# Статус лабораторной «Transfer Learning» — 14.10.2026

**Опубликованный проект Google Teachable Machine:** https://teachablemachine.withgoogle.com/models/rPbqmrUgK/

Проверено автоматическим запуском GitHub Actions:
- Публичный `metadata.json` содержит классы `bird`, `cat`, `dog`.
- Публичный `model.json` и файл весов доступны по HTTP.
- Результат проверки: [results/teachable_machine_verification.json](../results/teachable_machine_verification.json).
- Параметры, подтверждённые скриншотом пользователя: **25 эпох, batch size 16, learning rate 0.001**.
- Обучение: **80 высококачественных изображений на класс**, 240 всего. Отложенная тестовая выборка: 20 изображений на класс, 60 всего.

**Независимое тестирование Teachable Machine выполнено 14.10.2026:** 60 ранее не виденных снимков (по 20 на класс), правильных предсказаний **57/60**, Test Accuracy **95.00%**, macro Precision **95.07%**, macro Recall **95.00%**, macro F1 **95.00%**, multiclass log-loss **0.09558**. Матрица ошибок (порядок bird/cat/dog): `[[20,0,0],[0,18,2],[0,1,19]]`.

Подтверждающие файлы: [полные метрики](../results/teachable_machine_test_metrics.json), [предсказания по каждому изображению](../results/teachable_machine_test_predictions.csv), [матрица ошибок](../results/teachable_machine_confusion_matrix.png). Автоматический [успешный запуск GitHub Actions](https://github.com/mythical-crypto/university-notes/actions/runs/38575698894).

**Другая, самостоятельная модель:** MobileNetV2 на CIFAR-10; проверенные Accuracy **86.30%**, macro F1 **86.27%** на 3000 тестовых примерах. Источник: [results/metrics.json](../results/metrics.json), [матрица ошибок](../results/confusion_matrix.png). Нельзя переносить эти метрики на Teachable Machine.

**Подготовленный датасет для Teachable Machine:** [скачать ZIP](../results/dataset_100_per_class.zip) и [документация об источниках](../results/TEACHABLE_HIGHRES_DATASET.md).

Веб-проект Teachable Machine создан, обучен, опубликован и протестирован. Развёртывание в Yandex Datasphere и публикация на Yandex Disk пока не подтверждены.
