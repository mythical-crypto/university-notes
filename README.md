# Лабораторная работа: Computer Vision / Transfer Learning

**Дисциплина:** Интеллектуальные системы · **Группа 131** · **Студент:** Тимофеев Роман Сергеевич

**Задача:** классификация птиц, кошек и собак (`bird`, `cat`, `dog`) из CIFAR-10 с использованием предварительно обученной MobileNetV2 (ImageNet).

## Состав

- `Лабораторная_Transfer_Learning_CIFAR10.ipynb` — главный ноутбук, 4 этапа лабораторной, включая построение датасета, две модели, оценку метрик, тест и экспорт.
- `train_transfer_learning.py` — эквивалентный Python-скрипт.
- `demo_transfer_learning.py` — локальный Gradio-интерфейс после обучения.
- `Отчет_Transfer_Learning.docx` / `.pdf` — пояснительный отчет и бланк.
- Артефакты `metrics.json`, `confusion_matrix.png`, `main_learning.png`, `mobilenetv2_cifar3.h5` и другие генерируются *при запуске*, не включены без подтвержденного обучения.

## Изображения для Teachable Machine — обновлено 14.10.2026

- **[Скачать новый ZIP с качественными фотографиями](results/dataset_100_per_class.zip)** — 300 изображений, по 100 на класс (`bird`, `cat`, `dog`), `train` 80 и `test` 20 на класс.
- Фото птиц взяты из [Caltech-UCSD Birds-200-2011](https://huggingface.co/datasets/cassiekang/cub200_dataset), фото кошек и собак — из [Cats-Dogs-Birds](https://huggingface.co/datasets/Sijuade/Cats-Dogs-Birds). Эти снимки **не CIFAR-10**, качество исходников выше, искусственного увеличения нет.
- Сведения об источниках: [`results/TEACHABLE_HIGHRES_DATASET.md`](results/TEACHABLE_HIGHRES_DATASET.md), внутри ZIP — `manifest.csv` с источниками.
- **Не смешивать результаты:** метрики Accuracy 86,30% и F1 86,27% относятся к другой модели, обученной на CIFAR-10. Их нельзя выдавать за результаты Teachable Machine на новом датасете.

## Опубликованная модель Teachable Machine

- **Публичный проект:** https://teachablemachine.withgoogle.com/models/rPbqmrUgK/
- Подтверждение доступности модели, метаданных и весов: [results/teachable_machine_verification.json](results/teachable_machine_verification.json).
- Подтверждены метки: bird, cat, dog. Из скриншота обучения пользователя: 25 эпох, batch 16, learning rate 0.001; по 80 фотографий для обучения на класс.
- Подготовленный высококачественный набор: [results/dataset_100_per_class.zip](results/dataset_100_per_class.zip), 80 изображений для обучения и 20 отложенных тестовых в каждом классе. Источники: [results/TEACHABLE_HIGHRES_DATASET.md](results/TEACHABLE_HIGHRES_DATASET.md).
- Внимание: **Accuracy 86,30% и F1 86,27%** относятся к **отдельной MobileNetV2, обученной на CIFAR-10**. Не приписывать эти результаты Teachable Machine. Для неё необходим независимый тест на 60 фотографиях.

## Google Colab

1. Откройте `.ipynb` в Google Colab, выберите «Среда выполнения → Запустить все».
2. Дождитесь скачивания официального CIFAR-10 и весов ImageNet. При наличии GPU выбирайте T4.
3. Ноутбук автоматически скачает архив `lab_results.zip` (включая ZIP с 300 изображениями для Teachable Machine, графики, оценки и модели).
4. Teachable Machine уже обучена и экспортирована: https://teachablemachine.withgoogle.com/models/rPbqmrUgK/ (25 эпох, batch 16, lr 0.001; 80 обучающих фото на класс).

## Разделение данных

| Сценарий | Train | Validation | Test |
|---|---:|---:|---:|
| Stage1 dataset: 100/класс | 240 (80/класс) | — | 60 (20/класс) |
| Stage2 обучаемая мини-модель | 180 (60/класс) | 60 (20/класс) | 60 (20/класс) |
| Stage3 главная модель | 3600 (1200/класс) | 900 (300/класс) | 3000 (1000/класс) |

Все элементы собственного тестового набора главной модели взяты из официального CIFAR-10 test, отдельно от train.

## Архитектура

`Input(96×96×3) → Augmentation → Rescale(-1..1) → MobileNetV2(ImageNet, frozen) → GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(3, softmax)`.

**Собственная модель:** Adam, lr=0.001, batch=32, 8 эпох max, early stopping, валидация 20% обучающей подвыборки.

**Мини-модель:** Adam, lr=0.001, batch=16, 25 эпох max, early stopping.

**Примечание:** здесь не заявлены непроверенные accuracy / F1, не выданы несуществующие ссылки на Teachable Machine / Datasphere / Yandex Disk. Все количественные результаты формируются исполняемым ноутбуком.