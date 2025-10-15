# Arknights Integrated Strategies Collectibles Identifier / 明日方舟集成戰略藏品辨識器

## Warning / 警告

* This project is not affiliated with, endorsed by, or authorized by the game developers or publishers (such as Hypergryph, Longcheng, Yostar, or Studio Montagne). This project involves web scraping, and **anyone using this project does so at their own risk**. The author makes **no guarantees** that using this project does not violate any terms of service or laws in your jurisdiction. The author also **does not guarantee** that the use of this project will not lead to personal data leaks. The author **assumes no responsibility** for any damage or loss resulting from the use of this project, and **offers no implied warranties** of any kind. This project is provided **for academic research and personal use only**; **commercial use or redistribution is strictly prohibited**.

* 本專案與遊戲開發商或發行商（如鷹角網路、龍成網路、悠星網路或蒙塔山工作室）無任何關聯，也未經其授權。本專案含有對PRTS的爬蟲，任何使用本專案的人士，須自行承擔一切風險，專案作者不保證使用本專案不會違反任何使用條款或任何地區的法律法規，專案作者不保證使用本專案不會造成任何個人資料的洩漏，專案作者不會負責任何因使用本專案而引致之損失，專案作者不會作出任何默示的擔保。本專案僅供學術研究與個人用途，禁止用於任何商業用途或轉售。

## Feature / 功能

* Based on the collectibles data at `prts.wiki`, this project automatically identify which collectibles appears, and output their **simplified Chinese name**, which is suitable at analyzing the end picture.

* 基於 `prts.wiki` 的藏品圖像與名稱，本專案會分析圖像中含有哪些藏品，並以簡體中文輸出，適合用來分析結局畫面。

* Based on location information, we will make a rough guess for the collectibles that could not be located.

* 基於位置資訊，我們會對未能找到的藏品，並給出不太準確的猜測。

## Prerequisites / 前置需求
* Ensure you have Google Chrome 133.0 or higher installed. / 請確保您的電腦已安裝 Google Chrome 133.0 或以上版本。
* Ensure you have Python 3.11 or higher installed. / 請確保您的電腦已安裝 Python 3.11 或以上版本。

## Getting Started / 使用方法

1. **Clone the Repository**: / **複製項目**：  
    Open a terminal and run: / 打開終端機後執行：  
    ```bash
    git clone https://github.com/kaihuang1122/Arknights-Collectibles-Identifier
    ```
2. **Navigate to the Project Directory**: / **進入專案目錄**：  
    ```bash
    cd Arknights-Collectibles-Identifier
    ```
3. **Create a Virtual Environment (Recommended)**: / **創建虛擬環境（推薦）**：
    ```bash
    python -m venv venv
    .\venv\Scripts\activate (windows)
    source venv/bin/activate (Unix or MacOS)
    ```
4. **Install Dependencies**: / **安裝套件**：  
    ```bash
    pip install -r requirements.txt
    ```
5. **Download Data**: / **下載資料**：

    Run `greb.py` to download data, with the following arguments. / 執行 `greb.py` 下載資料，以下是需要的參數
    ```bash
    usage: greb.py [-h] [--is1] [--is2] [--is3] [--is4] [--is5] [--is6]

    Collect collectibles data and pictures from PRTS IS pages.

    options:
    -h, --help  show this help message and exit
    --is1       Collect collectibles from #1: Ceobe's Fungimist
    --is2       Collect collectibles from #2: Phantom & Crimson Solitaire
    --is3       Collect collectibles from #3: Mizuki & Caerula Arbor
    --is4       Collect collectibles from #4: Expeditioner's Jǫklumarkar
    --is5       Collect collectibles from #5: Sarkaz's Furnaceside Fables
    --is6       Collect collectibles from #6: Sui's Garden of Grotesqueries
    ```
    Example: If you want to use it at Sarkaz's Furnaceside Fables, execute: / 舉例：如果你想使用於薩卡茲的無終期語，則執行：
    ```python
    python greb.py --is5
    ```


6. **Run the matching code**: / **執行辨識程式**：  
    ```bash
    python match.py --img_path <your-image-path> --IS <which-IS>
    ```

    You can add some arguments based on the following description / 您可以根據以下描述添加一些參數
    ```bash
    usage: match.py [-h] --img_path IMG_PATH [--IS IS] [--show] [--show-metrics]

    Match collectibles in a screenshot.

    options:
    -h, --help           show this help message and exit
    --img_path IMG_PATH  Directory containing collectible templates.
    --IS IS              IS version (default: Sui's Garden of Grotesqueries)
    --show               Show matching results.
    --show-metrics       Display detailed matching metrics.
    ```



## Future works

- Better UI and i18n
- 使用 Media Wiki API 進行爬蟲