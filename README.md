# Генерация описаний товаров через Qwen3-VL-8B-Instruct

Скрипт добавляет описания товаров одежды на основе анализа изображений с помощью модели `Qwen/Qwen3-VL-8B-Instruct` из 🤗 Transformers.

- **Модель**: `Qwen/Qwen3-VL-8B-Instruct`
- **Библиотеки**: `transformers`, `torch`, `Pillow`, `requests`
- **Результат**: `description`, `keywords`, `styles` (1–3 из заранее заданного пула)
- **Изображения**: автоматически скачиваются и кэшируются локально в `./images`

## Установка

```bash
pip install -r requirements.txt
```

> Для GPU/CPU PyTorch на Windows см. официальные инструкции по установке Torch. Если `pip install torch` падает, установите подходящее колесо с сайта PyTorch.

## Использование

### Быстрый старт

```bash
python add_descriptions.py
```

По умолчанию:
- Читает товары из `../zara_us_optimized_categories.json`
- Обрабатывает первые `100` товаров
- Сохраняет результат в `./zara_with_descriptions_qwen.json`
- Кэширует изображения в `./images`

### Аргументы

```bash
# Обработать другое количество
python add_descriptions.py --limit 50

# Своя директория для изображений
python add_descriptions.py --images-dir ./my_images

# Свои пути к файлам
python add_descriptions.py --input ../path/to/input.json --output ./out.json

# Включить flash attention 2 (если доступно в окружении)
python add_descriptions.py --attn-impl flash_attention_2
```

## Формат выходных данных

Каждый товар дополняется полями:
- `description`: подробное описание + в конце склеенные `Keywords` и `Styles`
- `keywords`: строка с ключевыми словами через запятую
- `styles`: массив из 1–3 стилей
- `images`: сохранена только первая ссылка (исходного товара)

## Ссылки

- Модель Qwen на Hugging Face: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)


