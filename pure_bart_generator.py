from transformers import BartForConditionalGeneration, BartTokenizer
import torch, textwrap

tok   = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").eval()

def generate(context, max_len=60):
    # BART는 <s> … </s> 형태를 선호하므로 그대로 인코딩
    inputs = tok(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,  # ✨ pad도 해주자!
        add_special_tokens=False  # ⚠️ 토크나이저가 자동으로 <s>와 </s>를 추가하지 않도록 설정
    )
    print(tok.convert_ids_to_tokens(inputs["input_ids"][0]))
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,           # 샘플링 기반
            temperature=0.9,
            top_p=0.95,
            no_repeat_ngram_size=3,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )[0]
    return tok.decode(out_ids, skip_special_tokens=True)

# 수동으로 특수 토큰을 추가 (BOS는 BART 생성기가 자동으로 처리)
ctx = (
    "<s>User: Hi, I just broke up with my girlfriend and feel terrible.</s>"
    "System: I'm really sorry to hear that. It must be painful. "
    "Do you want to talk about what happened?</s>"
    "User: I just don't know how to move on.</s>"
    "System: "  # 마지막 System: 뒤에 </s> 없음
)

print(textwrap.fill("CTX >>> " + ctx, 120))
print("\nGEN >>>", generate(ctx))
