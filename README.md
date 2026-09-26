# 🏋️ CV for Fit — Real-Time Fitness Form Analyzer & API Engine

<div align="center">

**المحرك الذكي لتحليل أداء التمارين المنزلية والتوجيه الصوتي التفاعلي في الوقت الحقيقي**

*An AI-powered real-time exercise form analysis, interactive voice coaching & API integration engine*

</div>

## 🧠 تفاصيل عمل الذكاء الاصطناعي والموديل المستخدم | Deep Tech: AI Model & DTW Details

يعتمد النظام بالكامل على تكنولوجيا هجينة تدمج بين تتبع الهيكل العظمي ثلاثي الأبعاد وخوارزمية مطابقة السلاسل الزمنية الحركية:

### 1. موديل تتبع المفاصل: Google MediaPipe Pose Landmarker
* **فكرة عمل الموديل**: يعتمد المحرك على نموذج **MediaPipe Pose** المطور من قِبل Google، وهو عبارة عن شبكة عصبية عميقة (Deep Neural Network) مدربة على ملايين الصور الرياضية والحركية.
* **كيفية التتبع**: يقوم الموديل بتحديد **33 مفصلاً حركياً (Landmarks)** في كامل الجسم بالوقت الحقيقي. كل مفصل يحتوي على إحداثيات ثلاثية الأبعاد:
  * `X`: الإحداثي الأفقي في الصورة.
  * `Y`: الإحداثي الرأسي في الصورة.
  * `Z`: العمق النسبي للمفصل عن مركز الورك (Hip center).
  * `Visibility`: درجة ثقة الموديل في رؤية المفصل (لمنع حساب مفاصل مخفية أو خارج الكادر).
* **معالجة تباين الأجسام**: لكي لا يتأثر النظام باختلاف أطوال المستخدمين، أو بعدهم عن الكاميرا، أو حجم الغرفة، يتم تحويل إحداثيات النقاط الخام (Raw Coordinates) إلى **زوايا مفاصل نسبية (Normalized Joint Angles)** باستخدام حساب المتجهات (Vector Math). الزاوية بين ثلاث مفاصل متصلة تظل ثابتة رياضياً بغض النظر عن المسافة وحجم الجسم.

### 2. خوارزمية التعلم والمطابقة الزمنية: Dynamic Time Warping (DTW)
* **المشكلة الحركية**: البشر يؤدون التمارين بسرعات متفاوتة؛ فقد يؤدي المدرب عدة السكوات في 4 ثوانٍ، بينما يؤديها المستخدم في 6 ثوانٍ أو ينزل ببطء ويصعد بسرعة. المقارنة التقليدية إطاراً تلو الآخر (Frame-by-Frame) ستفشل تماماً بسبب هذا التفاوت الزمني.
* **الحل (DTW)**: هي خوارزمية ذكاء اصطناعي لمطابقة السلاسل الزمنية ذات السرعات المختلفة. تقوم الخوارزمية بعمل **"تمديد أو ضغط زمني" (Time Warping)** لتجد أفضل مسار للمطابقة (Optimal Warp Path) بين حركة المستخدم وحركة المدرب.
* **سرعة الأداء (fastDTW)**: نستخدم النسخة التقريبية السريعة **fastDTW** والتي تحول التعقيد الحسابي من تعقيد تربيعي $O(N^2)$ إلى تعقيد خطي $O(N)$، مما يسمح بحساب التطابق فور انتهاء العدة مباشرة على الأجهزة الضعيفة والهواتف دون أي تأخير.
* **المقارنة الحركية**:
  1. يقوم المدرب بتسجيل "الحركة المرجعية" عبر الكاميرا لمرة واحدة.
  2. يقوم النظام بحفظ مصفوفة الزوايا الزمنية للمدرب كبصمة حركية ثنائية الأبعاد في ملف `.npy`.
  3. أثناء تمرين المستخدم، يسجل النظام الزوايا الخاصة به من بداية العدة لنهايتها.
  4. تطبق الخوارزمية مسافة إقليدس (Euclidean Distance) التراكمية بين السلسلتين لحساب متوسط الانحراف بالدرجات، ثم يتم تحويل هذا الانحراف إلى درجة مئوية (Score) تعبر عن مدى مطابقة أداء المستخدم للمحترفين.

---

## 🏃 التمارين المنزلية الـ 8 المدعومة | Supported Exercises

يدعم النظام 8 تمارين منزلية أساسية تغطي كافة العضلات، ومقسمة حسب زاوية التصوير المطلوبة لضمان أعلى دقة تتبع:

### 📸 أ. تمارين الرؤية الأمامية (Front View Exercises)
في هذه التمارين، يواجه المتدرب الكاميرا بالكامل. يقوم النظام بحساب الزوايا الرياضية **ثلاثية الأبعاد (3D Vectors)** للجانبين الأيمن والأيسر معاً ويحسب المتوسط لهما لتفادي تشوهات المنظور وتوفير قراءات دقيقة ومتماثلة.

#### 1. 👐 Shoulder Press (الكتف)
- **المفاصل المقاسة**: الكوع (Shoulder-Elbow-Wrist)، الكتف (Hip-Shoulder-Elbow).
- **التوجيه التفاعلي**:
  - عدم تماثل حركة الذراعين (Symmetry check): *"Keep your arms symmetrical. Press both weights together."*
  - نزول الأكواع لأسفل بشكل زائد: *"Do not drop your elbows too low. Keep them at 90 degrees."*
  - عدم فرد الأذرع بالكامل في الأعلى: *"Press all the way up! Extend your arms overhead."*

---

### 📸 ب. تمارين الرؤية الجانبية (Side View Exercises)
في هذه التمارين، يقف المتدرب بالجنب للكاميرا. يتعرف النظام تلقائياً على الجانب الأقرب والمناسب للرؤية الجانبية (Auto Side Detection) ويقوم بعمل إسقاط رياضي ثنائي الأبعاد (2D Planar Projection) لتقليل الضوضاء الناتجة عن العمق.

#### 2. 🦵 Squat (السكوات)
- **المفاصل المقاسة**: الركبة (Hip-Knee-Ankle)، الفخذ (Shoulder-Hip-Knee)، الكاحل (Knee-Ankle-Foot).
- **التوجيه التفاعلي**:
  - لو لم يصل لعمق متوازي مع الأرض: *"Go a little deeper! Try to bring your thighs parallel to the ground."*
  - لو مال الصدر للأمام بشكل زائد: *"Keep your chest up. Avoid leaning too far forward."*
  - خروج الركبة عن أصابع القدم (Knee Travel check): *"Keep your weight on your heels. Don't let your knees travel too far forward."*
  - عند إنهاء الحركة بدون فرد كامل: *"Stand up completely and squeeze your glutes at the top."*

#### 3. 💪 Bicep Curl (البايسيبس)
- **المفاصل المقاسة**: الكوع (Shoulder-Elbow-Wrist) ، الكتف (Hip-Shoulder-Elbow).
- **التوجيه التفاعلي**:
  - أرجحة الذراع وتحرك الكوع: *"Keep your upper arm still. Don't swing."*
  - ميل الجسم للخلف (Lean check): *"Stand up straight and engage your core. Don't lean back."*
  - عدم ثني الكوع كاملاً: *"Curl higher!"*
  - عدم فرد الكوع كاملاً في الأسفل: *"Extend your arm fully."*

#### 4. 👟 Lunge (الطعن)
- **المفاصل المقاسة**: الركبة الأمامية (Hip-Knee-Ankle)، الفخذ (Shoulder-Hip-Knee).
- **التوجيه التفاعلي**:
  - خروج الركبة الأمامية عن أصابع القدم (Knee Forward check): *"Do not push your front knee too far forward. Keep it above your ankle."*
  - عمق النزول: *"Step deeper into the lunge. Lower your back knee."*

#### 5. 🧘 Push-up (الضغط)
- **المفاصل المقاسة**: الكوع (Shoulder-Elbow-Wrist)، الكتف (Hip-Shoulder-Elbow)، الفخذ (Shoulder-Hip-Knee).
- **التوجيه التفاعلي**:
  - تقوس أو ارتخاء الظهر (Hip Sag/Spike): *"Keep your body straight. Do not sag or spike your hips."*
  - النزول غير الكافي: *"Go lower! Try to bring your chest closer to the floor."*

#### 6. 🦸 Superman (سوبرمان - الظهر السفلي)
- **المفاصل المقاسة**: الفخذ (Shoulder-Hip-Knee)، الكتف (Elbow-Shoulder-Hip).
- **التوجيه التفاعلي**:
  - رفع الصدر والفخذين عن الأرض: *"Lift your chest and thighs higher off the ground."*

#### 7. 🪑 Chair Dips (الترايسيبس)
- **المفاصل المقاسة**: الكوع (Shoulder-Elbow-Wrist)، الكتف (Hip-Shoulder-Elbow)، الفخذ (Shoulder-Hip-Knee).
- **التوجيه التفاعلي**:
  - عمق النزول: *"Dip lower! Try to flex your elbows to 90 degrees."*
  - بعد الجسم عن المقعد أو الكرسي: *"Keep your back close to the bench or chair."*

#### 8. 🛑 Plank (البلانك - تمرين الثبات)
- **المفاصل المقاسة**: الفخذ (Shoulder-Hip-Knee)، الكتف (Hip-Shoulder-Elbow)، الركبة (Hip-Knee-Ankle).
- **حساب التكرار**: يتم احتساب تكرار واحد لكل **5 ثوانٍ** من الثبات بالوضعية الصحيحة.
- **التوجيه التفاعلي**:
  - نزول الحوض والبطن لأسفل: *"Keep your hips straight. Avoid sagging or spiking your hips."*
  - رفع المؤخرة لأعلى: *"Bring your hips down. Your body should form a straight line."*

---

## 🛠️ البنية التحتية والمخدم API Server (FastAPI + Supabase)

لتشغيل النظام كمخدم سحابي يستقبل بث الحركة من تطبيق الهواتف الذكية (Flutter) ويخزن النتائج في **Supabase**:

### 1. هيكل المخدم
توجد الملفات في المجلد `app/`:
- [`app/main.py`](file:///d:/college/semester%208/Graduation%20project/cv%20for%20fit/app/main.py): الكود الأساسي للمخدم، مسارات الويب الـ HTTP، ومستقبل تدفق المفاصل بالـ WebSocket.
- [`app/exercises.py`](file:///d:/college/semester%208/Graduation%20project/cv%20for%20fit/app/exercises.py): مشغل معالجة الفريمات الفردية بدون الحاجة لفتح واجهات عرض رسومية في المخدم.
- [`app/dtw_evaluator.py`](file:///d:/college/semester%208/Graduation%20project/cv%20for%20fit/app/dtw_evaluator.py): مسؤل جلب وتحميل تسلسل الفيديوهات المرجعية للتطابق الزمني الديناميكي.
- [`app/database.py`](file:///d:/college/semester%208/Graduation%20project/cv%20for%20fit/app/database.py): موصل قاعدة البيانات بـ Supabase مع دعم تخزين محلي (SQLite fallback) لضمان عدم توقف النظام عند انقطاع الإنترنت أو عدم توافر المفاتيح.

### 2. مسارات المخدم الـ API Endpoints
- **HTTP GET `/api/exercises`**: للحصول على قائمة تفاصيل الـ 8 تمارين المنزلية المدعومة.
- **HTTP POST `/api/session/complete`**: لحفظ تقرير جلسة تمرين كاملة.
  - *المدخلات*:
    ```json
    {
      "session_id": "uuid-string",
      "user_id": "user-uuid-from-supabase",
      "exercise_type": "squat",
      "rep_count": 12,
      "avg_score": 88.5
    }
    ```
- **WebSocket `/ws/workout`**: لبث حركة المفاصل في الوقت الحقيقي والحصول على التوجيه المباشر.
  - *المعاملات المطلوبة*: `?exercise=squat&userId=user_id_here`
  - *الرسالة المرسلة من تطبيق الهاتف (فريم بعد فريم)*:
    ```json
    {
      "landmarks": [
        {"x": 0.52, "y": 0.35, "z": -0.1, "visibility": 0.99},
        ...
      ]
    }
    ```
  - *الرد التفاعلي المستمر (في الوقت الحقيقي)*:
    ```json
    {
      "rep_count": 4,
      "phase": "WORKING",
      "active_side": "both",
      "angles": {"hip": 124.2, "knee": 110.5, "ankle": 95.8},
      "orientation": "front",
      "coaching_tip": "Push your knees outwards. Do not let them cave in.",
      "new_rep": false
    }
    ```

---

## 📹 نظام تسجيل مرجع الفيديو | Register Reference System

يتيح ملف [`register_reference.py`](file:///d:/college/semester%208/Graduation%20project/cv%20for%20fit/register_reference.py) للمسؤولين تحويل أي فيديو مسجل لمدرب محترف يؤدي التمرين بشكل مثالي إلى ملف مرجعي `.npy` لتعتمد عليه خوارزمية الـ DTW.

### كيفية تسجيل مرجع تمرين جديد:
قم بتجهيز فيديو بصيغة `.mp4` لأداء صحيح ومثالي، ثم نفذ الأمر التالي:
```bash
py register_reference.py --video "path/to/expert_video.mp4" --exercise "squat"
```
سيقوم البرنامج بتحليل المفاصل وإخراج ملف مرجع الحركة وحفظه تلقائياً في المسار `app/references/squat_reference.npy`.

---

## 🚀 التشغيل المحلي | How to Run Locally

### 1. تثبيت الحزم
قم بتثبيت جميع الحزم اللازمة:
```bash
pip install fastapi uvicorn supabase python-dotenv mediapipe opencv-python numpy fastdtw scipy pyttsx3
```

### 2. تشغيل المخدم (FastAPI)
لتشغيل مخدم الـ API والـ WebSockets محلياً للربط مع Flutter:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. تشغيل واجهة الفحص الرسومية المحلية (مع الصوت التفاعلي)
اضغط ضغطاً مزدوجاً على **`run.bat`** لاختيار التمرين من قائمة الـ 8 تمارين وتشغيل الكاميرا محلياً لمشاهدة رسومات الهيكل العظمي والـ HUD والاستماع للمدرب الصوتي.

---

## 📊 نظام التقييم وتطابق الحركة (DTW Score)
النظام يقوم بقياس مدى جودة الأداء بالمقارنة الرياضية الزمنية.
يتم تقييم الجودة باستخدام المعادلة اللوجستية الأسية لتوفير حساسية مريحة للمستخدم:
$$\text{Score} = 100 \times e^{-0.022 \times \text{Distance}_{\text{Normalized}}}$$

- **🟢 80 – 100**: أداء ممتاز جداً ومطابق للمدرب.
- **🟡 55 – 79**: أداء متوسط مقبول مع تحذيرات لضبط التوازن والسرعة.
- **🔴 0 – 54**: أداء غير دقيق يتطلب التباطؤ والتركيز في حركة المفاصل.
