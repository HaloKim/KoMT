import json
import asyncio
import argparse
import re
import ast
from typing import List, Dict, Any, Optional
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from dataclasses import dataclass
from pathlib import Path
import time
import logging
import openai
from tqdm import tqdm
from openai import AsyncOpenAI

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KoMTQuestion:
    question_id: int
    category: str
    turns: List[str]
    reference: List[str] = None

@dataclass
class ModelResponse:
    question_id: int
    model_name: str
    responses: List[str]
    completion_time: float

@dataclass
class JudgePrompt:
    name: str
    type: str
    system_prompt: str
    prompt_template: str
    output_format: str

class KoMTJudge:
    def __init__(self, openai_api_key: str, judge_model: str = "gpt-4"):
        """GPT 심사위원 초기화"""
        self.judge_model = judge_model
        # 새로운 OpenAI 클라이언트 초기화 방법
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
    async def call_openai_api(self, messages: List[Dict], temperature: float = 0) -> str:
        """OpenAI API 호출"""
        try:
            response = await self.client.chat.completions.create(
                model=self.judge_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                timeout=60
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            return ""
    
    def build_conversation_for_judge(self, question: KoMTQuestion, response: ModelResponse) -> str:
        """동적으로 N턴 대화 구성"""
        conversation = "<|어시스턴트 A와 사용자와의 대화 시작|>\n\n"
        
        # 모든 턴에 대해 동적으로 처리
        for i, (turn_question, turn_answer) in enumerate(zip(question.turns, response.responses)):
            conversation += f"### 사용자:\n{turn_question}\n\n"
            conversation += f"### 어시스턴트 A:\n{turn_answer}\n\n"
        
        conversation += "<|어시스턴트 A와 사용자와의 대화 종료|>"
        return conversation
    
    def build_judge_prompt(self, question: KoMTQuestion, response: ModelResponse) -> tuple:
        """동적 심사 프롬프트 구성"""
        turn_count = len(question.turns)
        
        if turn_count == 1:
            # 단일 턴
            system_prompt = """공정한 심사위원이 되어 사용자 질문에 대해 인공지능 어시스턴트가 제공한 아래 답변의 품질을 평가해 주세요. 평가할 때는 답변의 유용성, 관련성, 정확성, 깊이, 창의성 및 대답의 자세한 정도와 같은 요소를 고려해야 합니다. 간단한 설명을 제공하는 것으로 평가를 시작하세요. 가능한 한 객관적으로 평가합니다. 설명을 제공한 후에는 다음 형식을 엄격하게 준수하여 답변을 1점부터 10점까지 평가해야 합니다: "[[rating]]", 예시: "점수: [[5]]"."""
            
            user_prompt = f"""[사용자 질문]
{question.turns[0]}

[어시스턴트 답변 시작]
{response.responses[0] if response.responses else ""}
[어시스턴트 답변 끝]"""
        
        else:
            # 멀티 턴 (N턴)
            system_prompt = f"""공정한 심사위원이 되어 사용자 질문에 대해 인공지능 어시스턴트가 제공한 아래 답변의 품질을 평가해 주세요. 평가할 때는 답변의 유용성, 관련성, 정확성, 깊이, 창의성, 일관성 및 대답의 자세한 정도와 같은 요소를 고려해야 합니다. 이 대화는 총 {turn_count}턴으로 이루어져 있으며, 전체 대화의 흐름과 마지막 턴의 답변 품질에 초점을 맞춰 평가해야 합니다. 간단한 설명을 제공하는 것으로 평가를 시작하세요. 가능한 한 객관적으로 평가합니다. 설명을 제공한 후에는 다음 형식을 엄격하게 준수하여 답변을 1점부터 10점까지 평가해야 합니다: "[[rating]]", 예시: "점수: [[5]]"."""
            
            user_prompt = self.build_conversation_for_judge(question, response)
        
        return system_prompt, user_prompt
    
    async def judge_single_response(self, question: KoMTQuestion, response: ModelResponse) -> Dict:
        """단일 응답 평가"""
        system_prompt, user_prompt = self.build_judge_prompt(question, response)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        judgment = await self.call_openai_api(messages)
        
        # 점수 추출
        score = self.extract_score(judgment)
        
        return {
            "question_id": question.question_id,
            "model": response.model_name,
            "judge": self.judge_model,
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": len(question.turns)
        }
    
    def extract_score(self, judgment: str) -> int:
        """판정에서 점수 추출"""
        # [[숫자]] 패턴 찾기
        pattern = r"\[\[(\d+\.?\d*)\]\]"
        match = re.search(pattern, judgment)
        
        if match:
            try:
                score = float(match.group(1))
                return int(round(score))
            except:
                return -1
        
        # 백업 패턴들
        backup_patterns = [
            r"점수\s*:\s*\[\[(\d+\.?\d*)\]\]",
            r"평점\s*:\s*\[\[(\d+\.?\d*)\]\]",
            r"점수\s*:\s*(\d+\.?\d*)",
            r"평점\s*:\s*(\d+\.?\d*)",
            r"(\d+)\s*점",
            r"(\d+\.?\d*)\s*/\s*10"
        ]
        
        for pattern in backup_patterns:
            match = re.search(pattern, judgment)
            if match:
                try:
                    score = float(match.group(1))
                    return int(round(min(10, max(1, score))))
                except:
                    continue
                    
        return -1
    
    async def judge_all_responses(self, questions: List[KoMTQuestion], 
                                 responses: List[ModelResponse], 
                                 batch_size: int = 5) -> List[Dict]:
        """모든 응답 평가"""
        # 질문 ID로 매핑
        question_map = {q.question_id: q for q in questions}
        
        judgments = []
        
        # 배치 단위로 처리
        for i in tqdm(range(0, len(responses), batch_size), desc="GPT-4 평가 중"):
            batch = responses[i:i + batch_size]
            
            # 배치 내 모든 응답을 비동기로 평가
            tasks = []
            for response in batch:
                if response.question_id in question_map:
                    question = question_map[response.question_id]
                    tasks.append(self.judge_single_response(question, response))
            
            if tasks:
                batch_judgments = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 예외 처리 및 결과 수집
                for judgment in batch_judgments:
                    if isinstance(judgment, Exception):
                        logger.error(f"판정 오류: {judgment}")
                    else:
                        judgments.append(judgment)
        
        return judgments

class KoMTEvaluator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """KoMT 벤치마크 평가기 초기화"""
        self.model_path = model_path
        
        # vLLM AsyncEngine 설정
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=8192
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=8192,
            stop=None
        )
        
    def load_questions(self, question_file: str) -> List[KoMTQuestion]:
        """komt.jsonl 파일에서 질문 로드"""
        questions = []
        
        with open(question_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                question = KoMTQuestion(
                    question_id=data['question_id'],
                    category=data['category'],
                    turns=data['turns'],
                    reference=data.get('reference', None)
                )
                questions.append(question)
                
        logger.info(f"총 {len(questions)}개 질문 로드 완료")
        return questions
    
    def build_conversation_prompt(self, turns: List[str]) -> str:
        """멀티턴 대화를 하나의 프롬프트로 구성 (동적 N턴 지원)"""
        if len(turns) == 1:
            return f"사용자: {turns[0]}\n\n조수:"
        
        conversation = ""
        for i, turn in enumerate(turns):
            if i == 0:
                conversation += f"사용자: {turn}\n\n조수:"
            else:
                # 이전 턴의 응답이 완료된 후 다음 턴 추가
                conversation += f"\n\n사용자: {turn}\n\n조수:"
                
        return conversation
    
    async def generate_response(self, question: KoMTQuestion) -> ModelResponse:
        """각 턴을 순차적으로 처리"""
        start_time = time.time()
        
        responses = []
        conversation_history = ""
        
        for turn_idx, turn_question in enumerate(question.turns):
            # 현재까지의 대화 히스토리 구성
            if turn_idx == 0:
                prompt = f"사용자: {turn_question}\n\n조수:"
            else:
                # 이전 대화 포함하여 현재 턴 추가
                prompt = conversation_history + f"\n\n사용자: {turn_question}\n\n조수:"
            
            # 현재 턴에 대한 응답 생성
            request_id = f"request_{question.question_id}_turn_{turn_idx}"
            
            results_generator = self.engine.generate(
                prompt,
                self.sampling_params,
                request_id=request_id
            )
            
            # 응답 수집
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            # 생성된 응답 추출 (조수: 이후 부분만)
            generated_text = final_output.outputs[0].text.strip()
            
            # 응답에서 불필요한 부분 제거
            if "사용자:" in generated_text:
                generated_text = generated_text.split("사용자:")[0].strip()
            
            responses.append(generated_text)
            
            # 대화 히스토리 업데이트
            conversation_history = prompt + generated_text
            
            logger.info(f"턴 {turn_idx + 1} 완료: {generated_text[:100]}...")
        
        completion_time = time.time() - start_time
        
        return ModelResponse(
            question_id=question.question_id,
            model_name=self.model_path,
            responses=responses,
            completion_time=completion_time
        )
    
    def parse_multi_turn_response(self, generated_text: str, expected_turns: int) -> List[str]:
        """생성된 텍스트를 턴별로 분리 (개선된 버전)"""
        # 디버깅을 위한 로그
        logger.info(f"파싱할 텍스트: {generated_text[:200]}...")
        
        # "사용자:"를 기준으로 분리
        parts = generated_text.split("사용자:")
        
        responses = []
        
        if len(parts) > 1:
            # 첫 번째 응답 (첫 번째 "조수:" 이후부터 첫 번째 "사용자:" 이전까지)
            first_part = parts[0].strip()
            responses.append(first_part)
            
            # 나머지 응답들
            for i in range(1, min(len(parts), expected_turns)):
                part = parts[i]
                # "조수:" 이후 부분 찾기
                if "조수:" in part:
                    response_text = part.split("조수:", 1)[1].strip()
                    # 다음 턴이 있다면 그 이전까지만 취하기
                    if i < len(parts) - 1:
                        response_text = response_text.split("\n\n사용자:")[0].strip()
                    responses.append(response_text)
                else:
                    responses.append("")
        else:
            # 파싱 실패 시 전체를 첫 번째 응답으로 처리
            responses.append(generated_text.strip())
        
        # 부족한 턴 수를 빈 문자열로 채우기
        while len(responses) < expected_turns:
            responses.append("")
        
        # 로그로 파싱 결과 확인
        for i, response in enumerate(responses[:expected_turns]):
            logger.info(f"턴 {i+1} 응답: {response[:100]}...")
        
        return responses[:expected_turns]
    
    async def evaluate_all(self, questions: List[KoMTQuestion], batch_size: int = 10) -> List[ModelResponse]:
        """모든 질문에 대해 배치 처리로 평가"""
        all_responses = []
        
        # 배치 단위로 처리
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            logger.info(f"배치 {i//batch_size + 1} 처리 중... ({len(batch)}개 질문)")
            
            # 배치 내 모든 질문을 비동기로 처리
            tasks = [self.generate_response(question) for question in batch]
            batch_responses = await asyncio.gather(*tasks)
            
            all_responses.extend(batch_responses)
            
            logger.info(f"배치 완료. 진행률: {len(all_responses)}/{len(questions)}")
            
        return all_responses
    
    def save_results(self, responses: List[ModelResponse], output_file: str):
        """결과를 jsonl 형식으로 저장"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for response in responses:
                result = {
                    'question_id': response.question_id,
                    'model_name': response.model_name,
                    'responses': response.responses,
                    'completion_time': response.completion_time
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        logger.info(f"결과가 {output_file}에 저장되었습니다.")

def save_judgments(judgments: List[Dict], output_file: str):
    """판정 결과 저장"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for judgment in judgments:
            f.write(json.dumps(judgment, ensure_ascii=False) + '\n')
    
    logger.info(f"판정 결과가 {output_file}에 저장되었습니다.")

def print_evaluation_summary(questions: List[KoMTQuestion], judgments: List[Dict]):
    """평가 요약 출력 (턴 분포 포함)"""
    # 턴별 통계
    turn_stats = {}
    turn_scores = {}
    
    for judgment in judgments:
        turn_count = judgment['turn']
        if turn_count not in turn_stats:
            turn_stats[turn_count] = 0
            turn_scores[turn_count] = []
        
        turn_stats[turn_count] += 1
        if judgment['score'] != -1:
            turn_scores[turn_count].append(judgment['score'])
    
    # 카테고리별 통계
    question_map = {q.question_id: q for q in questions}
    category_stats = {}
    category_scores = {}
    
    for judgment in judgments:
        q_id = judgment['question_id']
        if q_id in question_map:
            category = question_map[q_id].category
            if category not in category_stats:
                category_stats[category] = 0
                category_scores[category] = []
            
            category_stats[category] += 1
            if judgment['score'] != -1:
                category_scores[category].append(judgment['score'])
    
    print("\n" + "="*60)
    print("KoMT 벤치마크 평가 요약")
    print("="*60)
    
    # 전체 평균 점수
    all_scores = [j['score'] for j in judgments if j['score'] != -1]
    if all_scores:
        print(f"전체 평균 점수: {sum(all_scores)/len(all_scores):.2f}")
        print(f"유효한 평가: {len(all_scores)}/{len(judgments)}")
    
    print("\n턴별 통계:")
    print("-" * 40)
    for turn_count in sorted(turn_stats.keys()):
        count = turn_stats[turn_count]
        scores = turn_scores[turn_count]
        avg_score = sum(scores)/len(scores) if scores else 0
        print(f"{turn_count}턴: {count}개 질문, 평균 점수: {avg_score:.2f}")
    
    print("\n카테고리별 통계:")
    print("-" * 40)
    for category in sorted(category_stats.keys()):
        count = category_stats[category]
        scores = category_scores[category]
        avg_score = sum(scores)/len(scores) if scores else 0
        print(f"{category}: {count}개 질문, 평균 점수: {avg_score:.2f}")
    
    print("="*60)

async def main():
    parser = argparse.ArgumentParser(description="KoMT 벤치마크 평가")
    parser.add_argument("--model-path", type=str, required=True, help="모델 경로")
    parser.add_argument("--question-file", type=str, default="komt.jsonl", help="질문 파일 경로")
    parser.add_argument("--output-file", type=str, default="komt_results.jsonl", help="출력 파일 경로")
    parser.add_argument("--judgment-file", type=str, default="komt_judgments.jsonl", help="판정 파일 경로")
    parser.add_argument("--batch-size", type=int, default=10, help="배치 크기")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="텐서 병렬 처리 크기")
    parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API 키")
    parser.add_argument("--judge-model", type=str, default="gpt-4", help="심사 모델")
    parser.add_argument("--skip-generation", action="store_true", help="응답 생성 건너뛰고 기존 결과로 평가")
    
    args = parser.parse_args()
    
    # 질문 로드
    questions = []
    with open(args.question_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = KoMTQuestion(
                question_id=data['question_id'],
                category=data['category'],
                turns=data['turns'],
                reference=data.get('reference', None)
            )
            questions.append(question)
    
    responses = []
    
    if not args.skip_generation:
        # 평가기 초기화
        evaluator = KoMTEvaluator(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size
        )
        
        # 응답 생성
        logger.info("모델 응답 생성 시작...")
        responses = await evaluator.evaluate_all(questions, batch_size=args.batch_size)
        
        # 결과 저장
        evaluator.save_results(responses, args.output_file)
    else:
        # 기존 결과 로드
        logger.info("기존 결과 파일 로드 중...")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                response = ModelResponse(
                    question_id=data['question_id'],
                    model_name=data['model_name'],
                    responses=data['responses'],
                    completion_time=data['completion_time']
                )
                responses.append(response)
    
    # GPT-4 판정
    logger.info("GPT-4 판정 시작...")
    judge = KoMTJudge(openai_api_key=args.openai_api_key, judge_model=args.judge_model)
    judgments = await judge.judge_all_responses(questions, responses)
    
    # 판정 결과 저장
    save_judgments(judgments, args.judgment_file)
    
    # 요약 출력
    print_evaluation_summary(questions, judgments)

if __name__ == "__main__":
    asyncio.run(main())