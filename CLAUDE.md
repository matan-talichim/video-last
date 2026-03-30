# AI Editor — CLAUDE.md (חוקת הפרויקט)

## מה זה הפרויקט
עורך וידאו אוטומטי מבוסס AI.
שני מצבים: (1) עריכת חומרי גלם קיימים, (2) יצירת סרטון מאפס מפרומפט.
השפה העיקרית: עברית. ממשק: GUI (React) + API server.

## כלל עליון: מחקר לפני יישום
לפני כתיבת קוד לרכיב חדש:
1. חקור את הדרך הכי מודרנית ויציבה
2. בדוק תאימות ספריות
3. בחר את הדרך הפשוטה והבדוקה
4. וודא שגרסאות יציבות
5. אם לא בטוח — שאל

## חוקי ברזל — לא לשבור לעולם

### 1. מודולריות מוחלטת
- כל יכולת (פיצ'ר) היא **step נפרד** בתיקייה `server/src/pipeline/steps/`
- כל step מייצא פונקציה אחת שמקבלת `StepContext` ומחזירה `StepResult`
- step **לעולם לא** מייבא מ-step אחר. תקשורת רק דרך ה-context
- step חדש = קובץ חדש. לא מוסיפים לוגיקה ל-step קיים

### 2. אל תיגע בליבה
- הקבצים הבאים הם **קריטיים**. לא לשנות בלי אישור מפורש:
  - `server/src/pipeline/engine.ts`
  - `server/src/pipeline/types.ts`
  - `server/src/index.ts`
  - `server/src/utils/logger.ts`
  - `client/src/App.tsx`
  - `client/src/i18n/index.ts`
- לפני כל שינוי בקבצים האלה: **הסבר מה אתה רוצה לשנות ולמה, וחכה לאישור**

### 3. שלב אחד בכל פעם
- לא לבנות 2 פיצ'רים במקביל
- לא להוסיף קוד "לעתיד" או "להכנה"
- כל PR/commit עושה דבר אחד בלבד
- אחרי כל שינוי — לבדוק שהכל עובד מקצה לקצה

### 4. הגדרות בקונפיג, לא בקוד
- כל ערך שיכול להשתנות → ב-`server/config/default.json`
- **לא** hardcode ערכים בקוד

### 5. לוגים תמיד
- כל step חייב לכתוב ללוג: מה קיבל, מה עשה, כמה זמן לקח, מה החזיר
- **לעולם לא** `console.log` — רק `logger.info()` / `logger.error()` / `logger.debug()`

### 6. טיפול בשגיאות
- כל step עטוף ב-try/catch
- step שנכשל **לא** מפיל את כל ה-pipeline — ממשיך לstep הבא
- שגיאות FFmpeg חייבות לכלול את הפקודה המלאה שנכשלה

### 7. קבצים זמניים
- כל קובץ זמני נוצר בתיקייה `temp/` בתוך תיקיית ה-output
- בסוף ה-pipeline — ניקוי אוטומטי של `temp/`
- שמות קבצים זמניים כוללים timestamp

### 8. עברית ו-RTL
- ממשק משתמש: עברית ברירת מחדל, עם אפשרות לאנגלית
- כל טקסט UI עובר דרך מערכת i18n (react-i18next) — לא hardcode טקסט
- כיוון (RTL/LTR) משתנה אוטומטית לפי השפה הנבחרת
- שמות קבצים ומשתנים — באנגלית

### 9. Git
- commit message באנגלית, קצר וברור
- פורמט: `feat: add X` / `fix: resolve Y` / `refactor: simplify W`
- לא לעשות commit של node_modules, temp files, או קבצי וידאו

### 10. כשיש ספק — תשאל
- אם משימה לא ברורה → שאל לפני שאתה כותב קוד
- אם שינוי עלול להשפיע על רכיבים אחרים → תציין את זה
- אם יש כמה דרכים — הצג אותן ותן המלצה

---

## טכנולוגיות

| רכיב | טכנולוגיה |
|-------|-----------|
| Backend (`server/`) | Node.js 20+, TypeScript, Express, ESM |
| Frontend (`client/`) | React 18, TypeScript, Vite, Tailwind CSS 3 |
| i18n | react-i18next (עברית + אנגלית) |
| וידאו | FFmpeg 6+ |
| רנדר | Remotion (גל 3+) |
| תמלול | TBD — ייקבע בגל 1 |
| AI Brain | Claude API (גל 2) |
| קריינות | ElevenLabs (גל 3) |
| מוזיקה | Suno (גל 3) |
| B-Roll | KIE.ai (גל 3) |

## Pipeline

קובץ וידאו → [Step 1] → [Step 2] → ... → [Step 7: Edit Assembly] → קובץ תוצאה

### Step 7 — Edit Assembly
- FFmpeg: חיתוך keep segments, padding 50ms, fade 30ms, concat → `edited.mp4`
- approve מפעיל edit אוטומטית

## גלי בנייה

- **גל 0** ← כאן — שלד: pipeline, API, GUI, i18n, FFmpeg
- **גל 1** עיבוד בסיסי — תמלול, הסרת שתיקות, כתוביות
- **גל 2** ניתוח — hooks, quotes, pacing
- **גל 3** יצירת תוכן — B-Roll, קריינות, מוזיקה
- **גל 4** עריכה מתקדמת — beat sync, VFX, color
- **גל 5** תוצרים — versions, revisions, export
- **גל 6** מתקדם — brand kit, templates, dubbing
- **גל 7** לימוד — self-improving system

## סטטוס משימות

| # | משימה | סטטוס | תיאור |
|---|-------|-------|-------|
| 7 | Edit Assembly | ✅ | חיתוך והרכבה FFmpeg (padding, fade, concat -c copy, ResultPage עם נגן) |

## מבנה תיקיות (עיקרי)

```
server/src/pipeline/steps/
  └── edit-assembly.ts        # Step 7 — חיתוך והרכבה FFmpeg

client/src/pages/
  └── ResultPage.tsx           # דף תוצאה עם נגן וידאו

<output>/
  └── edited.mp4               # קובץ תוצאה סופי
```
