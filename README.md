# Лабораторная работа: Computer Vision / Transfer Learning

**Дисциплина:** Интеллектуальные системы · **Группа 131** · **Студент:** Тимофеев Роман Сергеевич

**Задача:** классификация птиц, кошек и собак (`bird`, `cat`, `dog`) из CIFAR-10 с использованием предварительно обученной MobileNetV2 (ImageNet).

## Состав

- `Лабораторная_Transfer_Learning_CIFAR10.ipynb` — главный ноутбук, 4 этапа лабораторной, включая построение датасета, две модели, оценку метрик, тест и экспорт.
- `train_transfer_learning.py` — эквивалентный Python-скрипт.
- `demo_transfer_learning.py` — локальный Gradio-интерфейс после обучения.
- `Отчет_Transfer_Learning.docx` / `.pdf` — пояснительный отчет и бланк.
- Артефакты `metrics.json`, `confusion_matrix.png`, `main_learning.png`, `mobilenetv2_cifar3.h5` и другие генерируются *при запуске*, не включены без подтвержденного обучения.

## Google Colab

1. Откройте `.ipynb` в Google Colab, выберите «Среда выполнения → Запустить все».
2. Дождитесь скачивания официального CIFAR-10 и весов ImageNet. При наличии GPU выбирайте T4.
3. Ноутбук автоматически скачает архив `lab_results.zip` (включая ZIP с 300 изображениями для Teachable Machine, графики, оценки и модели).
4. В Teachable Machine внешний проект создаётся вручную: `https://teachablemachine.withgoogle.com/train/image`. Загрузить `bird`, `cat`, `dog` из `dataset_100_per_class.zip`, задать `epochs=25`, `batch=16`, `learning_rate=0.001`, экспортировать URL.

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