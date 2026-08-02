<!--
  Chainlit renders this on the EMPTY chat screen, above the starters defined in
  chainlit_app.py. It must live here rather than in an on_chat_start message:
  sending any message makes the thread non-empty, and Chainlit then draws
  neither the README nor the starters. Verified on the deployed app -- the
  starters were registered correctly server-side (/project/settings returned all
  four) and were simply never drawn, because a welcome message had taken the
  screen.

  The article count below is HARDCODED and will go stale. Update it when the
  corpus grows past the current 94 indexed articles (see DEFERRED_TODO.md); the
  live number is always in /health and under the Retrieval step of any answer.
-->

# Բարև Ձեզ 👋

Ես պատասխանում եմ **infocom.am**-ի **94 վերլուծական հոդվածի** հիման վրա։
Գրեք հայերեն, անգլերեն կամ ռուսերեն։

## Ի՞նչ թեմաներ են ընդգրկված

- դատաիրավական համակարգ, քրեական վարույթներ, արդարադատություն
- ընտրություններ, կուսակցություններ, խորհրդարան
- պետական գնումներ և կոռուպցիոն ռիսկեր
- սոցիալական քաղաքականություն՝ կենսաթոշակներ, առողջապահություն
- տրանսպորտ և քաղաքային ենթակառուցվածքներ
- գիտություն, կրթություն, թվային իրավունքներ

## Ինչպես եմ աշխատում

- պատասխանում եմ **միայն** այս հոդվածների բովանդակությամբ և միշտ նշում եմ
  աղբյուրը՝ [1], [2]
- եթե հոդվածներում տվյալները չկան, ուղիղ կասեմ՝ փոխանակ ենթադրելու
- ամսաթվերը կարևոր են. հոդվածները ժամանակի մեջ սահմանափակ են, ուստի նոր
  իրադարձությունները կարող են ընդգրկված չլինել
- ցանկացած պատասխանի տակ բացեք **Retrieval** / **Generation**՝ օգտագործված
  հատվածները, միավորները և արժեքը տեսնելու համար

Ներքևի օրինակներից ընտրեք մեկը կամ գրեք Ձեր հարցը։
