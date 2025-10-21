import ollama

def extract_keywords(prompt: str):
    system_prompt = (
    "Bạn là trợ lý AI chuyên trích xuất từ khóa tiếng Việt. "
    "Nhiệm vụ của bạn là trích xuất **mọi danh từ xuất hiện trong câu**, "
    "bao gồm cả các danh từ chính và danh từ phụ trong cụm danh từ. "
    "Nếu trong câu có cụm như 'nhẫn vàng', bạn phải trích xuất cả 'nhẫn' và 'vàng'. "
    "Không lấy động từ, tính từ, đại từ hay các từ loại khác. "
    "Không được dịch, không được suy diễn, không được thêm hoặc bỏ sót từ. "
    "Kết quả trả về chỉ gồm danh sách các danh từ tách riêng, cách nhau bằng dấu phẩy."
    )




    response = ollama.chat(model='qwen2.5:1.5b', messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu: {prompt}\nHãy trích xuất từ khóa:"}
    ])

    content = response["message"]["content"]
    keywords = [k.strip().lower() for k in content.split(",") if k.strip()]
    return keywords
