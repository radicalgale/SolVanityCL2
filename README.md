# Solana Vanity GPU Generator

Утилита для высокопроизводительного поиска «красивых» Solana‑адресов (vanity addresses) на GPU под управлением OpenCL (CUDA).

---

## 🚀 Возможности

- Генерация и проверка **16 777 216** ключей за один батч (при `--iteration-bits=24`) на каждом GPU.
- Поддержка **множественных префиксов** (`--starts-with A --starts-with B …`) и суффикса (`--ends-with`).
- Сохранение найденных адресов в:
    - **CSV** (`results.csv`)

    - **JSON** (каждый ключ в отдельном файле `found/JSON/<pubkey>.json`, содержит массив 64 байта)
- Удобная настройка через CLI‑флаги или один `config.json`.

---

## 🔧 Требования

- **Python 3.8+**
- NVIDIA GPU с поддержкой OpenCL / CUDA
- Установленные драйверы NVIDIA + OpenCL runtime
- Windows 10+ / Linux / macOS

---

## 📦 Установка

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/radicalgale/SolVanityCL2.git
   cd SolVanityCL2
   ```

2. Установите зависимости:
   ```bash
   python -m pip install -r requirements.txt
    ```
---

## 📂 Структура проекта 
```SolVanityCL/
├── core/
│   ├── cli.py              # Основной CLI
│   ├── config.py           # DEFAULT_ITERATION_BITS, HostSetting
│   ├── opencl/
│   │   └── manager.py      # Выбор и инициализация GPU
│   ├── searcher.py         # Searcher + multi_gpu_init
│   └── utils/
│       ├── helpers.py      # load_kernel_source(), check_character()
│       └── crypto.py       # save_keypair(), get_public_key_from_private_bytes()
├── opencl/
│   └── kernel.cl           # OpenCL‑ядро generate_pubkey
├── main.py                 # Точка входа
└── config.json             # Пример конфигурации
```
---

## ⚙️ Конфигурация через JSON

```
{
  "startsWith": ["ABC", "123", "FOO"],   // массив префиксов
  "endsWith": "",                        // строка (пустая, если не нужен суффикс)
  "count": 10,                           // сколько совпадений искать (0 = бесконечно)
  "iterationBits": 24,                   // 2^24 = 16 777 216 ключей за батч
  "caseSensitive": true,                 // учёт регистра
  "outputDir": "./found",                // куда сохранять результаты
  "selectDevice": false,                 // вручную выбирать GPU
  "recordAll": false                     // логировать каждый стартовый seed
}
```
---

## ▶️ Запуск

1. Через config.json:
   ```bash
   python main.py search-pubkey --config config.json
   ```
2. Без конфига, через CLI‑флаги:
    ```bash
    python main.py search-pubkey \
    --starts-with SVM --starts-with SOL \
    --ends-with XYZ \
    --count 5 \
    --iteration-bits 24 \
    --is-case-sensitive \
    --output-dir ./found \
    --record-all
    ```
3. Показать доступные GPU
    ```bash
    python main.py show-device
   ```
---

## 📂 Результаты

При --output-dir ./found структура будет такой:

```
found/
├── JSON/                 # JSON‑файлы с seed||pubkey (64 байта)
│   ├── <pub1>.json
│   └── <pub2>.json
└── results.csv           # publicKey,privateKey (Base58 seed+pub)
```

---

## 💡 Советы и рекомендации
### Регулировка batch‑size:
Параметр ```--iteration-bits``` N задаёт 2^N ключей за батч.
Если батч выходит дольше 2 с (Windows TDR), уменьшите N или добавьте кратковременную паузу.

### Температура и троттлинг:
Мониторьте ```nvidia-smi -l 1```. Держите hotspot <100 °C, лучше <90 °C.

### Использование префиксов:
Добавление 3−5 префиксов влияет на скорость ≪1 % (основная нагрузка — крипто).

### Максимальная производительность:
Удалите встроенный ```sleep``` в ```Searcher.find()```, если не боитесь TDR и троттлинга, и подберите ```iteration_bits``` для батчей <2 с.

---

## 📝 Лицензия
### Based on https://github.com/WincerChan/SolVanityCL
### Build better by SVMURAI
### Revamped by radical_gale