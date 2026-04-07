"""
Скрипт для добавления описаний товаров через Amazon Nova Pro (AWS Bedrock).
Использует модель amazon.nova-pro-v1:0 для анализа изображений и генерации
ключевых слов, стилей и подробного текстового описания предмета одежды.

Threading: параллельная обработка + немедленная запись в JSONL-файл.
Resume: при повторном запуске пропускает уже обработанные товары.
S3 checkpoint: периодически синхронизирует результаты в S3 (для resume при рестарте пода).
"""

import json
import os
import time
import requests
import hashlib
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Set
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


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
    """Генерирует описания товаров через Amazon Nova Pro (AWS Bedrock)."""

    def __init__(
        self,
        images_dir: Optional[str] = None,
        aws_profile: Optional[str] = None,
        region: str = "us-east-1",
    ):
        self.style_pool = STYLE_POOL

        if images_dir:
            self.images_dir = Path(images_dir)
        else:
            self.images_dir = Path(__file__).parent / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Поддерживает как AWS профиль, так и переменные окружения
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile, region_name=region)
            creds_label = f"профиль: {aws_profile}"
        else:
            session = boto3.Session(region_name=region)
            creds_label = "переменные окружения (AWS_ACCESS_KEY_ID / IAM Role)"

        # Каждый поток должен иметь свой клиент — boto3 не thread-safe
        self._session = session
        self._region = region
        self._local = threading.local()
        print(f"[INFO] Bedrock клиент ({creds_label}, регион: {region})")
        print(f"[INFO] Изображения: {self.images_dir}")

    def _get_client(self):
        """Возвращает thread-local Bedrock клиент."""
        if not hasattr(self._local, "client"):
            self._local.client = self._session.client("bedrock-runtime", region_name=self._region)
        return self._local.client

    def _get_local_image_path(self, image_url: str) -> Path:
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        ext = ".jpg"
        lower = image_url.lower()
        if ".png" in lower:
            ext = ".png"
        elif ".webp" in lower:
            ext = ".webp"
        return self.images_dir / f"{url_hash}{ext}"

    @staticmethod
    def _normalize_url(url: str) -> str:
        u = (url or "").strip()
        return "https:" + u if u.startswith("//") else u

    def _load_image(self, image_url: str) -> Optional[bytes]:
        image_url = self._normalize_url(image_url)
        local_path = self._get_local_image_path(image_url)

        if local_path.exists():
            try:
                return local_path.read_bytes()
            except Exception:
                pass

        referer = "https://www.lamoda.ru/" if "lmcdn.ru" in image_url.lower() else "https://www.zara.com/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Referer": referer,
        }
        try:
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            image_data = b"".join(chunk for chunk in response.iter_content(8192) if chunk)
            local_path.write_bytes(image_data)
            return image_data
        except Exception as e:
            print(f"[ERROR] Не удалось скачать {image_url[:60]}: {e}")
            return None

    @staticmethod
    def _detect_media_type(data: bytes) -> str:
        if len(data) < 12:
            return "image/jpeg"
        if data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    def generate_description(self, product: Dict, image_url: str) -> Optional[Dict]:
        image_data = self._load_image(image_url)
        if not image_data:
            return None

        category = product.get("category", "unknown")
        name = product.get("name", "")
        color = product.get("color", "")
        media_type = self._detect_media_type(image_data)

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
                            {"image": {"format": media_type.split("/")[1], "source": {"bytes": image_b64}}},
                            {"text": prompt},
                        ],
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.8,
                },
            }

            client = self._get_client()
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            text = response_body["output"]["message"]["content"][0]["text"].strip()

            # Убираем markdown-обёртку если есть
            for prefix in ("```json", "```"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if "{" in text and "}" in text:
                text = text[text.find("{"):text.rfind("}") + 1]

            parsed = json.loads(text)
            keywords = parsed.get("keywords", "")
            styles = parsed.get("styles", [])
            item_desc = parsed.get("item_description", "")
            full_description = f"{item_desc} Keywords: {keywords} Styles: {', '.join(styles)}"
            return {"keywords": keywords, "styles": styles, "description": full_description}

        except json.JSONDecodeError as e:
            print(f"[ERROR] Не удалось распарсить JSON ответ: {e}")
            return None
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                print(f"[WARN] Bedrock throttling — пауза 10с...")
                time.sleep(10)
            else:
                print(f"[ERROR] Bedrock ошибка ({error_code}): {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Ошибка при генерации описания: {e}")
            return None


def s3_parse(s3_uri: str):
    """Разбирает s3://bucket/key в (bucket, key)."""
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def s3_download(s3_uri: str, local_path: Path, region: str = "us-east-1") -> bool:
    """Скачивает файл из S3, возвращает True если скачан."""
    try:
        bucket, key = s3_parse(s3_uri)
        s3 = boto3.client("s3", region_name=region)
        s3.download_file(bucket, key, str(local_path))
        print(f"[S3] Скачан checkpoint: {s3_uri} → {local_path}")
        return True
    except Exception as e:
        print(f"[S3] Checkpoint не найден или ошибка: {e}")
        return False


def s3_upload(local_path: Path, s3_uri: str, region: str = "us-east-1") -> bool:
    """Загружает файл в S3."""
    try:
        bucket, key = s3_parse(s3_uri)
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(str(local_path), bucket, key)
        print(f"[S3] Checkpoint сохранён: {local_path} → {s3_uri}")
        return True
    except Exception as e:
        print(f"[S3] Ошибка загрузки в S3: {e}")
        return False


def s3_sync_loop(local_path: Path, s3_uri: str, region: str, interval_sec: int, stop_event: threading.Event):
    """Фоновый поток: синхронизирует файл в S3 каждые interval_sec секунд."""
    while not stop_event.wait(interval_sec):
        if local_path.exists():
            s3_upload(local_path, s3_uri, region)
    # Финальная синхронизация при завершении
    if local_path.exists():
        s3_upload(local_path, s3_uri, region)


def load_processed_ids(output_file: Path) -> Set[str]:
    """Загружает уже обработанные ID из JSONL-файла (для resume при перезапуске)."""
    processed = set()
    if not output_file.exists():
        return processed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("id") or obj.get("url")
                if pid:
                    processed.add(str(pid))
            except json.JSONDecodeError:
                pass
    return processed


def process_products(
    input_file: str,
    output_file: str,
    limit: int = 0,
    images_dir: Optional[str] = None,
    aws_profile: Optional[str] = None,
    region: str = "us-east-1",
    workers: int = 5,
    s3_checkpoint: Optional[str] = None,
    s3_sync_interval: int = 120,
):
    """
    Обрабатывает товары параллельно (ThreadPoolExecutor).
    Каждый результат сразу пишется в JSONL-файл (одна строка = один товар).
    При повторном запуске пропускает уже обработанные товары (resume).
    s3_checkpoint: если указан (s3://bucket/key), скачивает при старте и синхронизирует каждые s3_sync_interval сек.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    # S3 checkpoint: скачать предыдущий прогресс
    if s3_checkpoint and not output_path.exists():
        s3_download(s3_checkpoint, output_path, region)

    print(f"[INFO] Загрузка товаров из {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        all_products = json.load(f)
    print(f"[INFO] Всего товаров: {len(all_products)}")

    # Resume: пропускаем уже обработанные
    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        print(f"[INFO] Уже обработано (resume): {len(processed_ids)} товаров")

    products = [
        p for p in all_products
        if str(p.get("id") or p.get("url")) not in processed_ids
    ]
    if limit and limit > 0:
        products = products[:limit]

    print(f"[INFO] Осталось обработать: {len(products)} товаров")
    if not products:
        print("[INFO] Все товары уже обработаны.")
        return

    total = len(products)
    describer = NovaImageDescriber(images_dir=images_dir, aws_profile=aws_profile, region=region)

    # Thread-safe счётчики и файловый замок
    write_lock = threading.Lock()
    counter_lock = threading.Lock()
    success_count = 0
    fail_count = 0

    def process_one(product: Dict) -> None:
        nonlocal success_count, fail_count

        images = product.get("images", [])
        if not images:
            with counter_lock:
                fail_count += 1
            return

        image_url = images[0]
        description_data = describer.generate_description(product, image_url)

        with counter_lock:
            done = success_count + fail_count + 1

        if description_data:
            result = product.copy()
            result["images"] = [image_url]
            result["description"] = description_data["description"]
            result["keywords"] = description_data["keywords"]
            result["styles"] = description_data["styles"]

            # Немедленно пишем в файл
            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success_count += 1

            print(f"[{done}/{total}] ✓ {product.get('name', '')[:50]}")
        else:
            with counter_lock:
                fail_count += 1
            print(f"[{done}/{total}] ✗ {product.get('name', '')[:50]}")

    print(f"[INFO] Запуск {workers} потоков...")
    start_time = time.time()

    # Фоновая синхронизация в S3
    stop_sync = threading.Event()
    if s3_checkpoint:
        sync_thread = threading.Thread(
            target=s3_sync_loop,
            args=(output_path, s3_checkpoint, region, s3_sync_interval, stop_sync),
            daemon=True,
        )
        sync_thread.start()
        print(f"[S3] Авто-синхронизация каждые {s3_sync_interval}с → {s3_checkpoint}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, p) for p in products]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Необработанное исключение в потоке: {e}")

    stop_sync.set()
    if s3_checkpoint:
        sync_thread.join(timeout=30)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Готово за {elapsed/3600:.1f} ч ({elapsed:.0f} сек)")
    print(f"  Успешно: {success_count}")
    print(f"  Ошибок:  {fail_count}")
    print(f"  Файл:    {output_path} (JSONL — одна строка = один товар)")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Генерация описаний товаров через Amazon Nova Pro (AWS Bedrock)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Входной JSON файл")
    parser.add_argument("--output", type=str, required=True, help="Выходной JSONL файл (append, resume-safe)")
    parser.add_argument("--limit", type=int, default=0, help="Лимит товаров (0 = все)")
    parser.add_argument("--workers", type=int, default=5, help="Кол-во параллельных потоков")
    parser.add_argument("--images-dir", type=str, default=None, help="Директория для кэша изображений")
    parser.add_argument("--profile", type=str, default=None, help="AWS профиль (по умолчанию: env vars / IAM Role)")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS регион")
    parser.add_argument("--s3-checkpoint", type=str, default=None, help="S3 URI для checkpoint (s3://bucket/key.jsonl)")
    parser.add_argument("--s3-sync-interval", type=int, default=120, help="Интервал синхронизации в S3 (сек, default: 120)")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Файл не найден: {args.input}")
        sys.exit(1)

    process_products(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        images_dir=args.images_dir,
        aws_profile=args.profile,
        region=args.region,
        workers=args.workers,
        s3_checkpoint=args.s3_checkpoint,
        s3_sync_interval=args.s3_sync_interval,
    )
