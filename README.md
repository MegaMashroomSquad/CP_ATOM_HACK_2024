# Система автоматизированного комплаенс-контроля документации

Этот проект реализует систему для автоматизированного анализа соответствия документации требованиям и стандартам, с возможностью определения уровня комплаенса по заданной шкале.

## Особенности

- Проводит качественный анализ документации на соответствие требованиям
- Определяет уровень соответствия по предоставленной шкале оценки
- Использует современные языковые модели для интеллектуального анализа текста
- Формирует структурированные отчеты в табличном формате
- Поддерживает как индивидуальный, так и пакетный анализ документов

## Компоненты

- `src/core.py`: Основной модуль анализа и классификации соответствия
- `src/model.py`: Реализация взаимодействия с языковыми моделями
- `src/scheme.py`: Структуры данных для хранения результатов анализа
- `src/timer.py`: Утилита измерения производительности
- `web_app.py`: Веб-интерфейс на Streamlit для удобной работы с системой
- `main.py`: Основной скрипт для запуска системы
- `FULL_LAUNCH.ipynb`: Полное руководство по запуску системы (глвное)

## Настройка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/CrudyLame/medznar-hack.git
   cd medznar-hack
   ```

2. Установите Poetry (если еще не установлен):
   ```bash
   pip install poetry
   ```

3. Установите зависимости проекта:
   ```bash
   poetry install
   ```

## Использование

1. Активируйте виртуальное окружение Poetry:
   ```bash
   poetry shell
   ```

2. Запустите приложение одним из способов:
   Через командную строку:
   ```bash
   python main.py
   ```

## Требования к системе

- Python 3.10 или выше
- Достаточно оперативной памяти для работы с языковыми моделями
- Доступ к интернету для загрузки моделей (при первом запуске)
