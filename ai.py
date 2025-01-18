from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

# FastAPI 애플리케이션 생성
app = FastAPI()

# KoGPT2 모델 및 토크나이저 로드
model_path = "./kogpt2-finetuned"
inference_model = GPT2LMHeadModel.from_pretrained(model_path)
inference_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

# Hugging Face pipeline 생성
generator = pipeline(
    "text-generation",
    model=inference_model,
    tokenizer=inference_tokenizer
)

# 입력 데이터 구조
class GenerateRequest(BaseModel):
    purpose: str  # 이주 목적
    priorities: str  # 우선순위

# 출력 데이터 구조
class GenerateResponse(BaseModel):
    generated_text: str  # 생성된 텍스트

# API 엔드포인트
@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    # 프롬프트 구성
    purpose = "삶의질"
    priorities = ["교통"]

    prompt = f"이주목적: {purpose}, 우선순위: {', '.join(priorities)}"

    # KoGPT2로 텍스트 생성
    output = generator(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    # 생성된 텍스트 추출
    generated_text = output[0]["generated_text"]

    # 반환
    return GenerateResponse(generated_text=generated_text)

# FastAPI 서버 실행
# uvicorn으로 실행 시 아래 부분은 생략 가능
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
