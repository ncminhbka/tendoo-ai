import json
import urllib.request
import ssl
import base64
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

class FluxClient:
    def __init__(self, api_url=None, auth_token=None, x_api_key=None, apikey=None, cookie=None):
        self.api_url = api_url or "https://ai-api.tendoo.vn/llm-gw/vllm/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {auth_token or 'sk-etX2FdIA1KB9qPZ8wh4uUA'}",
            "Content-Type": "application/json",
            "x-api-key": x_api_key or "63628873-b2b7-4c1d-a183-6eed5db78a00",
            "apikey": apikey or "nxhV1MEjmt4ge6IpKD76nhTzplUafhe/SlnunYq1wkU="
        }
        if cookie:
            self.headers["Cookie"] = cookie

    def generate_image(self, prompt: str, output_path: str = "output.png", 
                       reference_images: list = None,
                       width: int = 1024, height: int = 1024,
                       num_inference_steps: int = 5, 
                       guidance_scale: float = 1.0, 
                       strength: float = None,
                       seed: int = 42) -> str:
        """
        Gửi yêu cầu sinh/chỉnh sửa ảnh đến model FLUX.2-klein-9B.
        
        :param prompt: Mô tả văn bản
        :param output_path: Đường dẫn lưu ảnh kết quả
        :param reference_images: Danh sách các đường dẫn file ảnh tham chiếu (tối đa 10 ảnh)
        :param width: Chiều rộng ảnh (mặc định 1024)
        :param height: Chiều cao ảnh (mặc định 1024)
        :param num_inference_steps: Số bước lấy mẫu (khuyên dùng 4-5)
        :param guidance_scale: Độ bám sát prompt văn bản (1.0 - 4.0)
        :param strength: Độ tuân thủ cấu trúc ảnh gốc (0.1 = 100% tuân thủ, 1.0 = biến đổi hoàn toàn)
        :param seed: Seed cố định
        """
        content_payload = []
        
        # Thêm các ảnh tham chiếu vào content
        if reference_images:
            for img_item in reference_images:
                if os.path.exists(img_item):
                    with open(img_item, "rb") as f:
                        b64_img = base64.b64encode(f.read()).decode('utf-8')
                        data_uri = f"data:image/png;base64,{b64_img}"
                elif img_item.startswith("data:image") or img_item.startswith("http"):
                    data_uri = img_item
                else:
                    data_uri = f"data:image/png;base64,{img_item}"
                    
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri
                    }
                })
                
        # Thêm prompt văn bản
        content_payload.append({
            "type": "text",
            "text": prompt
        })

        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        extra_body_params = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height
        }
        
        if strength is not None:
            extra_body_params["strength"] = strength
            extra_body_params["denoising_strength"] = strength

        payload = {
            "model": "black-forest-labs/FLUX.2-klein-9B",
            "messages": [
                {
                    "role": "user",
                    "content": content_payload
                }
            ],
            "extra_body": extra_body_params
        }

        req = urllib.request.Request(
            self.api_url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=self.headers, 
            method='POST'
        )

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, context=ctx) as response:
            raw_res = response.read().decode('utf-8')
            data = json.loads(raw_res)
            
            msg_content = data["choices"][0]["message"]["content"]
            base64_str = None
            
            if isinstance(msg_content, list):
                for item in msg_content:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            img_url = item.get("image_url", {}).get("url", "")
                            base64_str = img_url.split(",", 1)[1] if "," in img_url else img_url
                        elif "image" in item:
                            img_val = item["image"]
                            base64_str = img_val.split(",", 1)[1] if "," in img_val else img_val
            elif isinstance(msg_content, str):
                try:
                    content_json = json.loads(msg_content)
                    img_val = content_json.get("image") or content_json.get("image_url")
                    if img_val:
                        base64_str = img_val.split(",", 1)[1] if "," in img_val else img_val
                except:
                    base64_str = msg_content.split(",", 1)[1] if "," in msg_content else msg_content

            if not base64_str:
                raise ValueError("Không tìm thấy dữ liệu ảnh Base64 trong response từ API.")

            img_bytes = base64.b64decode(base64_str)
            with open(output_path, "wb") as f:
                f.write(img_bytes)
                
            print(f"[OK] Saved image to {output_path} ({width}x{height}, {len(img_bytes)} bytes)")
            return output_path

if __name__ == "__main__":
    client = FluxClient()
    client.generate_image("Sinh ảnh cô gái đẹp", "test_out.png")
