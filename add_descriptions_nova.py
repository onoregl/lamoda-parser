"""
Скрипт для добавления описаний товаров через Amazon Nova Pro (AWS Bedrock)
Использует модель amazon.nova-pro-v1:0 для анализа изображений и генерации
ключевых слов, стилей и подробного текстового описания предмета одежды.
"""

import json
import os
import time
import requests
import hashlib
import re
import base64
from typing import Dict, Optional
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


# Стили с карточек AI Stylist: product_collections is_active=true
# (см. modera.fashion supabase/migrations/20260202_001_reduce_to_nine_styles.sql)
STYLE_POOL = [
    "casual",
    "elegant",
    "minimalistic",
    "streetwear",
    "old-money",
    "chic",
    "business-casual",
    "bohemian",
    "classic",
]

MODEL_ID = "amazon.nova-pro-v1:0"


class NovaImageDescriber:
    """Класс для генерации описаний товаров через Amazon Nova Pro (AWS Bedrock)"""

    def __init__(
        self,
        images_dir: Optional[str] = None,
        aws_profile: Optional[str] = "modera-gleb",
        region: str = "us-east-1",
    ):
        """
        Args:
            images_dir: Директория для локального хранения изображений
            aws_profile: Имя профиля ~/.aws/credentials; None = цепочка по умолчанию (AWS_* в env)
            region: AWS регион
        """
        self.style_pool = STYLE_POOL

        if images_dir:
            self.images_dir = Path(images_dir)
        else:
            script_dir = Path(__file__).parent
            self.images_dir = script_dir / "images"

        self.images_dir.mkdir(parents=True, exist_ok=True)

        if aws_profile is None:
            session = boto3.Session(region_name=region)
            creds_label = "переменные окружения (AWS_ACCESS_KEY_ID / ...)"
        else:
            session = boto3.Session(profile_name=aws_profile, region_name=region)
            creds_label = f"профиль: {aws_profile}"
        self.client = session.client("bedrock-runtime")
        print(f"[INFO] Инициализирован Bedrock клиент ({creds_label}, регион: {region})")

    def _get_image_filename(self, image_url: str) -> str:
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        ext = ".jpg"
        lower = image_url.lower()
        if ".png" in lower:
            ext = ".png"
        elif ".webp" in lower:
            ext = ".webp"
        return f"{url_hash}{ext}"

    def _get_local_image_path(self, image_url: str) -> Path:
        filename = self._get_image_filename(image_url)
        return self.images_dir / filename

    @staticmethod
    def _normalize_image_url(image_url: str) -> str:
        u = (image_url or "").strip()
        if u.startswith("//"):
            return "https:" + u
        return u

    def _download_image_with_headers(self, image_url: str, save_path: Optional[Path] = None) -> Optional[bytes]:
        try:
            referer = "https://www.zara.com/"
            if "lmcdn.ru" in image_url.lower() or "lamoda" in image_url.lower():
                referer = "https://www.lamoda.ru/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
                "Referer": referer,
            }
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            image_data = b""
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    image_data += chunk

            if save_path:
                save_path.write_bytes(image_data)

            return image_data
        except Exception as e:
            print(f"[ERROR] Не удалось скачать изображение {image_url}: {e}")
            return None

    def _load_image(self, image_url: str) -> Optional[bytes]:
        image_url = self._normalize_image_url(image_url)
        local_path = self._get_local_image_path(image_url)

        if local_path.exists():
            try:
                print(f"[INFO] Используем локальное изображение: {local_path.name}")
                return local_path.read_bytes()
            except Exception as e:
                print(f"[WARNING] Не удалось прочитать локальный файл {local_path}: {e}")

        print(f"[INFO] Скачиваем изображение: {image_url[:80]}...")
        image_data = self._download_image_with_headers(image_url, local_path)
        if image_data:
            print(f"[SUCCESS] Изображение сохранено: {local_path.name}")
        return image_data

    @staticmethod
    def _detect_image_media_type(image_data: bytes) -> str:
        """MIME по сигнатуре файла (Lamoda отдаёт WebP под .jpg в URL)."""
        if len(image_data) < 12:
            return "image/jpeg"
        if image_data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "image/webp"
        if image_data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        return "image/jpeg"

    def generate_description(self, product: Dict, image_url: str) -> Optional[Dict]:
        image_data = self._load_image(image_url)
        if not image_data:
            return None

        category = product.get("category", "unknown")
        name = product.get("name", "")
        color = product.get("color", "")
        media_type = self._detect_image_media_type(image_data)

        prompt = (
            f"Analyze this clothing item image. Focus ONLY on the clothing item itself, not the person wearing it.\n\n"
            f"Product information:\n"
            f"- Name: {name}\n"
            f"- Category: {category}\n"
            f"- Color: {color}\n\n"
            f"Tasks:\n"
            f"1. Describe the clothing item in detail (only the item, not the person). Include: material, texture, fit, design details, patterns, length, sleeves, closures, etc.\n"
            f"2. Extract 10-15 relevant keywords that describe the item (colors, materials, patterns, design elements, etc.)\n"
            f"3. From the style pool below, select 1-3 styles that best match this item: {', '.join(self.style_pool)}\n\n"
            f"IMPORTANT:\n"
            f"- Focus ONLY on the clothing item, ignore the person/model\n"
            f"- Consider the category \"{category}\" when describing\n"
            f"- Return your response in this exact JSON format:\n"
            "{\n"
            "  \"keywords\": \"keyword1, keyword2, keyword3, ...\",\n"
            "  \"styles\": [\"style1\", \"style2\"],\n"
            "  \"item_description\": \"Detailed description of the clothing item only\"\n"
            "}\n\n"
            "Return ONLY valid JSON, no additional text."
        )

        try:
            image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": media_type.split("/")[1],
                                    "source": {
                                        "bytes": image_b64,
                                    },
                                }
                            },
                            {
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.8,
                },
            }

            response = self.client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            text = response_body["output"]["message"]["content"][0]["text"].strip()

            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            if "{" in text and "}" in text:
                text = text[text.find("{"):text.rfind("}") + 1]

            try:
                parsed = json.loads(text)
                keywords = parsed.get("keywords", "")
                styles = parsed.get("styles", [])
                item_desc = parsed.get("item_description", "")
                full_description = f"{item_desc} Keywords: {keywords} Styles: {', '.join(styles)}"
                return {
                    "keywords": keywords,
                    "styles": styles,
                    "description": full_description,
                }
            except json.JSONDecodeError as e:
                print(f"[ERROR] Не удалось распарсить JSON ответ: {e}")
                print(f"[DEBUG] Ответ модели: {text[:500]}")
                return None

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            print(f"[ERROR] AWS Bedrock ошибка ({error_code}): {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Ошибка при генерации описания: {e}")
            import traceback
            traceback.print_exc()
            return None


def process_products(
    input_file: str,
    output_file: str,
    limit: int = 100,
    images_dir: Optional[str] = None,
    aws_profile: Optional[str] = "modera-gleb",
    region: str = "us-east-1",
    delay: float = 0.5,
):
    """
    Обрабатывает товары из JSON файла и добавляет описания через Nova Pro.
    Автоматически скачивает изображения в локальную директорию перед отправкой в модель.
    """
    input_path = str(input_file) if isinstance(input_file, Path) else input_file
    print(f"[INFO] Загрузка товаров из {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"[INFO] Найдено {len(products)} товаров")
    products_to_process = products[:limit]
    print(f"[INFO] Обрабатываем первые {len(products_to_process)} товаров")

    describer = NovaImageDescriber(images_dir=images_dir, aws_profile=aws_profile, region=region)

    print(f"[INFO] Изображения будут сохраняться в: {describer.images_dir}")

    processed_products = []
    failed_count = 0

    for i, product in enumerate(products_to_process, 1):
        print(f"\n[{i}/{len(products_to_process)}] Обработка: {product.get('name', 'Unknown')}")

        images = product.get("images", [])
        if not images:
            print(f"[SKIP] Нет изображений для товара {product.get('id')}")
            failed_count += 1
            continue

        image_url = images[0]
        print(f"[INFO] Анализ изображения: {image_url[:80]}...")

        description_data = describer.generate_description(product, image_url)

        if description_data:
            updated_product = product.copy()
            updated_product["images"] = [image_url]
            updated_product["description"] = description_data["description"]
            updated_product["keywords"] = description_data["keywords"]
            updated_product["styles"] = description_data["styles"]

            processed_products.append(updated_product)
            print(f"[SUCCESS] Описание добавлено")
            print(f"  Keywords: {description_data['keywords'][:80]}...")
            print(f"  Styles: {', '.join(description_data['styles'])}")
        else:
            print(f"[FAILED] Не удалось сгенерировать описание")
            failed_count += 1

        if i < len(products_to_process):
            time.sleep(delay)

    output_path = str(output_file) if isinstance(output_file, Path) else output_file
    print(f"\n[INFO] Сохранение результатов в {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_products, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Обработка завершена!")
    print(f"  Успешно обработано: {len(processed_products)}")
    print(f"  Ошибок: {failed_count}")
    print(f"  Результат сохранен в: {output_path}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Генерация описаний товаров через Amazon Nova Pro (AWS Bedrock)")
    parser.add_argument("--limit", type=int, default=100, help="Количество товаров для обработки (по умолчанию: 100)")
    parser.add_argument("--input", type=str, help="Путь к входному JSON файлу (по умолчанию: ../zara_us_optimized_categories.json)")
    parser.add_argument("--output", type=str, help="Путь к выходному JSON файлу (по умолчанию: zara_with_descriptions_nova.json)")
    parser.add_argument("--images-dir", type=str, help="Директория для локального хранения изображений (по умолчанию: ./images)")
    parser.add_argument("--profile", type=str, default="modera-gleb", help="AWS профиль из ~/.aws/credentials (по умолчанию: modera-gleb)")
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Не использовать профиль: брать креды из окружения (временные MFA: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)",
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS регион (по умолчанию: us-east-1)")
    parser.add_argument("--delay", type=float, default=0.5, help="Задержка между запросами в секундах (по умолчанию: 0.5)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.input:
        input_file = Path(args.input)
    else:
        input_file = project_root / "zara_us_optimized_categories.json"

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = script_dir / "zara_with_descriptions_nova.json"

    if not input_file.exists():
        print(f"[ERROR] Файл {input_file} не найден")
        print(f"[INFO] Текущая директория: {os.getcwd()}")
        sys.exit(1)

    aws_profile = None if args.no_profile else args.profile

    process_products(
        input_file,
        output_file,
        limit=args.limit,
        images_dir=args.images_dir,
        aws_profile=aws_profile,
        region=args.region,
        delay=args.delay,
    )
