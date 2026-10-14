# Лабораторная работа — Computer Vision / Transfer Learning

**Выполнил:** Тимофеев Роман Сергеевич · **Группа: 141**  
**Дисциплина:** Интеллектуальные системы  
**Тема:** Разработка системы компьютерного зрения для классификации изображений с использованием transfer learning.

## Преподавателю: с чего начать

**[Отчёт DOCX](01_Report/Отчет_Тимофеев_РС_141.docx)** · **[Отчёт PDF](01_Report/Отчет_Тимофеев_РС_141.pdf)** · **[Опубликованная Teachable Machine](https://teachablemachine.withgoogle.com/models/rPbqmrUgK/)**

| Каталог | Материалы | Что проверять |
|:--|:--|:--|
| [01_Report](01_Report/) | Отчёт Word, PDF, текст и встроенные иллюстрации | Описание всех четырёх этапов, таблица соответствия заданию |
| [02_Dataset](02_Dataset/) | 300 изображений птиц, кошек и собак в ZIP, 80/20, источники | 80 фото в train и 20 в test на класс |
| [03_Teachable_Machine](03_Teachable_Machine/) | Подтверждение публикации, метрики, 60 предсказаний и матрица ошибок | Тестовая Accuracy **95,00%** на 60 изображениях |
| [04_MobileNetV2](04_MobileNetV2/) | Обученная H5-модель, графики, показатели, отчёт по классам | Тестовая Accuracy **86,30%** на 3000 примерах CIFAR-10 |
| [05_Source_Code](05_Source_Code/) | Jupyter Notebook, Python-скрипты обучения и оценки | Воспроизводимость выполнения |
| [.github/workflows](.github/workflows/) | Сценарии GitHub Actions | Автоматическая проверка результатов |

## Подтверждённые результаты

| Эксперимент | Test | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|:--|--:|--:|--:|--:|--:|
| [Teachable Machine](https://teachablemachine.withgoogle.com/models/rPbqmrUgK/) | 60 новых фото | **95,00%** | 95,07% | 95,00% | 95,00% |
| MobileNetV2 / CIFAR-10 | 3000 | **86,30%** | 86,26% | 86,30% | 86,27% |

**Результаты нельзя напрямую сравнивать:** использованы разные изображения и разные объёмы тестовой выборки.

## Ссылки на реальные запуски

- [Обучение MobileNetV2 — success](https://github.com/mythical-crypto/university-notes/actions/runs/38549153693).
- [Проверка общедоступной Teachable Machine — success](https://github.com/mythical-crypto/university-notes/actions/runs/38575228344).
- [60 тестовых фотографий Teachable Machine — success](https://github.com/mythical-crypto/university-notes/actions/runs/38575698894).

## Что остаётся по буквальному тексту задания

Преподавательский бланк отдельно содержит Google Drive, Google Colab, Yandex Datasphere и Yandex Disk. Факт выполнения именно на этих платформах **не подтверждён**; в работе использованы GitHub, GitHub Actions и Teachable Machine. Конечные train accuracy/loss и длительность веб-обучения Teachable Machine не зафиксированы; независимые **test**-метрики подтверждены. Эти ограничения честно отмечены в отчёте.

**Сдача:** откройте DOCX/PDF, затем ссылку Teachable Machine, после — папки 03/04 с проверяемыми показателями. Всё лишнее и черновые заглушки из корня удалены.