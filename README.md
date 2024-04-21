# Що це таке?
Це програма, яка читає текст з вашого екрану. <br>

![Знімок екрана 2024-04-18 122325](https://github.com/Santabot123/ozvuchator/assets/56690519/8a5ba4d8-2d1d-4de9-8e11-69a19e0184da)

## Коротке демо
https://github.com/Santabot123/ozvuchator/assets/56690519/feeb47d6-1444-4b64-9a9c-d449d69d620a

## Демонстрація того як працює ручний режим
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/XBd4vCp3Rq4/0.jpg)](https://www.youtube.com/watch?v=XBd4vCp3Rq4)

## Демонстрація того як працює автоматичний режим
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NWhn4AtTWNg/0.jpg)](https://www.youtube.com/watch?v=NWhn4AtTWNg)


# Системні вимоги.
- OS: Windows 10/11 64-bit
- встановлений ffmpeg
- підключення до Інтернету

# Встановлення.
 Є 2 способи:
 1. exe-файл:
    - Встановіть ffmpeg і додайте його до PATH вашої системи (в інтернеті є багато гайдів, як це зробити)
    - Завантажте та розпакуйте цей [архів](https://drive.google.com/file/d/1dCeuGqOG6v4bfYHIggM--vihtEYOShJ4/view?usp=drive_link).
    - Потім для запуску перейдіть до `exe.win-amd64-3.9` і запустіть `ozvuchator.exe`.
 2. jupyter notebook:
    - Встановіть ffmpeg і додайте його до PATH вашої системи (в інтернеті є багато гайдів, як це зробити)
    - Встановіть [Anaconda](https://www.anaconda.com/installation-success?source=installer)
    - Створіть нове віртуальне середовище в Anaconda
    - Завантажте цей репозиторій і розархівуйте його, або скористайтеся `git clone https://github.com/Santabot123/ozvuchator`.
    - Відкрийте jupyter notebook, знайдіть місце, куди ви розархівували цей репозиторій, і відкрийте `ozvuchator.ipynb`.
    - Натисніть ⏩.  
    - Зачекайте (перший запуск буде довгим, оскільки потрібно завантажити всі необхідні бібліотеки)

# Використання
Існує 2 режими : ручний та автоматичний.
- Ручний режим - ви натискаєте кнопку активації (F2 за замовчуванням) і вибираєте область, після чого текст буде озвучений один раз.
- Автоматичний режим - ви обираєте область екрану, і коли в цій області з'являється новий текст, він буде автоматично озвучений.



# Зверніть увагу
- Щоб прочитати, що робить той чи інший параметр, наведіть на нього курсор
- Якщо у вас встановлена відеокарта Nvidia, за замовчуванням буде використовуватися вона, у всіх інших випадках буде використовуватися CPU
- Чим довший текст, тим більша затримка перед генерацією голосу
- Якщо ви збираєтеся використовувати цю програму під час гри, вам може знадобитися переключити гру у віконний/безрамочний режим
- Розпізнавання тексту буде добре працювати тільки коли використовуються друковані літери

# Список підтримуваних мов:
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



