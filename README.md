# Що це таке?
Це програма, яка читає текст з вашого екрану. Вона може читати записки, субтитри тощо. Вона також може перекладати текст перед тим, як він буде озвучений. 
Ось кілька прикладів того, як це виглядає і як звучить:

![Знімок екрана 2024-04-18 124413](https://github.com/Santabot123/ozvuchator/assets/56690519/1f905e47-afa7-4d1d-962b-c535d5eb15ec)

##  Коротке демо
https://github.com/Santabot123/ozvuchator/assets/56690519/2a3650de-e093-48bf-8d13-e77204468ee6

## Демонстрація того як працює ручний режим
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/XBd4vCp3Rq4/0.jpg)](https://www.youtube.com/watch?v=XBd4vCp3Rq4)

## Демонстрація того як працює автоматичний режим
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NWhn4AtTWNg/0.jpg)](https://www.youtube.com/watch?v=NWhn4AtTWNg)


# # Системні вимоги
- OS: Windows 10/11 64-bit
- підключення до Інтернету

# Встановлення
 There are 2 ways:
 Є 2 способи:
 1. exe-файл:
    - Виберіть версію (якщо ви не знаєте, яку вибрати, прочитайте розділ **Версії** нижче) 
    - Завантажте та розархівуйте один з цих архівів: [Pytesseract](https://drive.google.com/file/d/1oHwtpnOMQK-DUUiOAeekGwXaplZM-gS2/view?usp=sharing), [EasyOCR](https://drive.google.com/file/d/1JKMzWmEHGqZk0-FD3_ZalcqHBvNe2qZ9/view?usp=sharing), [EasyOCR+Pytesseract](https://drive.google.com/file/d/1oHwtpnOMQK-DUUiOAeekGwXaplZM-gS2/view?usp=sharing).
    - Для запуску перейдіть до `exe.win-amd64-3.10` і запустіть `ozvuchator.exe`.
 3. jupyter notebook:
   - Встановіть [Anaconda](https://www.anaconda.com/installation-success?source=installer)
    - Створіть нове віртуальне середовище у Anaconda
    - Завантажте цей репозиторій і розархівуйте його, або скористайтеся `git clone https://github.com/Santabot123/ozvuchator`.
    - Виберіть версію (якщо ви не знаєте, яку вибрати, прочитайте розділ **Версії** нижче) 
    - Якщо ви хочете використовувати Pyteeseract, завантажте [Tesseract 5.3.3](https://github.com/UB-Mannheim/tesseract/wiki) (можливо, коли ви читаєте цю інструкцію, буде новіша версія, але я не знаю, чи буде вона працювати так само добре, як 5.3.3, тому рекомендується використовувати 5.3.3) і встановіть її до папки Tesseract-OCR, розташованої у папці, куди ви завантажили цей репозиторій.
    - Відкрийте jupyter notebook, знайдіть місце, куди ви розпакували цей репозиторій, і відкрийте файл `ozvuchator.ipynb`, який відповідає вашій обраній версії.
    - Натисніть ⏩.  
    - Зачекайте (перший запуск буде довгим, оскільки вам потрібно завантажити всі необхідні бібліотеки)

# Версії

Існує три версії: Pytesseract, EasyOCR та Pytesseract+EasyOCR. Ось деяка інформація про них, щоб ви могли вирішити, яка з них вам потрібна:
- **Pytesseract** - використовує Tesseract, обирайте цю версію, якщо текст, який ви хочете озвучити: контрастує з фоном, виглядає як відсканований документ або скріншот. Переконайтеся, що обрана вами область екрану не містить елементів інтерфейсу/рамок вікон/іконок, оскільки вони можуть бути неправильно розпізнані як орфографічні символи. Ця версія також займає найменше місця на диску. Приклади тексту, з яким добре справляється Pytesseract:<br>
![Знімок екрана 2024-05-02 191519](https://github.com/Santabot123/ozvuchator/assets/56690519/9bf8da3b-6e42-4b44-974b-353b9caafa91)
![Знімок екрана 2024-05-02 192450](https://github.com/Santabot123/ozvuchator/assets/56690519/ca7f454c-750b-41e4-8d38-376f8f723980)
![Знімок екрана 2024-04-27 182902](https://github.com/Santabot123/ozvuchator/assets/56690519/6b93a6a2-fce8-4db0-b5f4-1c00ddd67ab3)

- **EasyOCR** - виберіть цю версію, якщо текст погано контрастує з фоном, на фотографіях або має деякі спотворення через перспективу. EasyOCR також може використовувати GPU Nvidia замість CPU. Приклади тексту, з яким EasyOCR добре справляється:
![Знімок екрана 2024-05-02 192758](https://github.com/Santabot123/ozvuchator/assets/56690519/12f55c92-5157-4258-a10e-70726d0d0afa)
![Знімок екрана 2024-05-02 193230](https://github.com/Santabot123/ozvuchator/assets/56690519/8e06752e-164b-4b3f-960f-e929dc2a4888)

- Pytesseract+EasyOCR - це, по суті, дві попередні версії, об'єднані в одну, що дає вам можливість вибирати, який метод використовувати.<br>
![Знімок екрана 2024-05-02 193531](https://github.com/Santabot123/ozvuchator/assets/56690519/fc241c2b-ce47-4092-bd57-ca4b74ab3e94)


# Використання
 
Існує 2 режими використання: ручний і автоматичний.
- Ручний режим - ви натискаєте кнопку активації (F2 за замовчуванням) і вибираєте область, після чого текст буде озвучений один раз.
- Автоматичний режим - ви обираєте область екрану, і коли в цій області з'являється новий текст, він буде автоматично озвучений.

Якщо ви вже натиснули кнопку Запустити, а потім хочете змінити налаштування, вам потрібно закрити Ozvuchator і відкрити його знову.



# Зверніть увагу
- Щоб дізнатися, що робить той чи інший параметр, наведіть на нього курсор.
- Чим довше речення, тим більша затримка перед відтворенням звуку.
- Якщо ви збираєтеся використовувати цю програму під час гри, вам може знадобитися переключити гру у віконний/безрамочний режим.
- Розпізнавання тексту буде добре працювати **тільки з друкованим текстом** .
<br>

# Список підтримуваних мов для версії що використовує EasyOCR:
- afrikaans : 'af'<br>
- albanian : 'sq'<br>
- arabic : 'ar'<br>
- bengali : 'bn'<br>
- bosnian : 'bs'<br>
- bulgarian : 'bg'<br>
- croatian : 'hr'<br>
- czech : 'cs'<br>
- danish : 'da'<br>
- dutch : 'nl'<br>
- english : 'en'<br>
- estonian : 'et'<br>
- filipino : 'tl'<br>
- french : 'fr'<br>
- german : 'de'<br>
- hindi : 'hi'<br>
- hungarian : 'hu'<br>
- icelandic : 'is'<br>
- indonesian : 'id'<br>
- italian : 'it'<br>
- japanese : 'ja'<br>
- kannada : 'kn'<br>
- korean : 'ko'<br>
- latin : 'la'<br>
- latvian : 'lv'<br>
- malay : 'ms'<br>
- marathi : 'mr'<br>
- nepali : 'ne'<br>
- norwegian : 'no'<br>
- polish : 'pl'<br>
- portuguese : 'pt'<br>
- romanian : 'ro'<br>
- slovak : 'sk'<br>
- spanish : 'es'<br>
- swahili : 'sw'<br>
- swedish : 'sv'<br>
- tamil : 'ta'<br>
- telugu : 'te'<br>
- thai : 'th'<br>
- turkish : 'tr'<br>
- ukrainian : 'uk'<br>
- urdu : 'ur'<br>
- vietnamese : 'vi'<br>

# Список підтримуваних мов для версії що використовує Pytesseract:
- Afrikaans : af
- Albanian : sq
- Arabic : ar
- Bengali : bn
- Bosnian : bs
- Bulgarian : bg
- Burmese : my
- Catalan : ca
- Croatian : hr
- Czech : cs
- Danish : da
- Dutch : nl
- English : en
- Estonian : et
- Finnish : fi
- French : fr
- German : de
- Gujarati : gu
- Hindi : hi
- Hungarian : hu
- Icelandic : is
- Indonesian : id
- Italian : it
- Japanese : ja 
- Kannada : kn
- Khmer : km
- Korean : ko
- Latin : la
- Latvian : lv
- Malay (macrolanguage) : ms
- Malayalam : ml
- Marathi : mr
- Modern Greek (1453-) : el
- Nepali (macrolanguage) : ne
- Norwegian : no
- Polish : pl
- Portuguese : pt
- Romanian : ro
- Serbian : sr
- Sinhala : si
- Slovak : sk
- Spanish : es
- Sundanese : su
- Swahili (macrolanguage) : sw
- Swedish : sv
- Tagalog : tl
- Tamil : ta
- Telugu : te
- Thai : th
- Turkish : tr
- Ukrainian : uk
- Urdu : ur
- Vietnamese : vi

