# Implementation Plan — Fitness CV Module Expansion, FastAPI & Supabase Integration (v2)

This plan details the addition of new exercises (total of 8 exercises covering all main muscle groups) and explains exactly how the reference video mechanism is registered and evaluated using Dynamic Time Warping (DTW).

---

## 1. How the Reference Video System Works

Instead of training complex deep learning models, we use a **Biomechanical Reference Matching** approach. Here is the step-by-step workflow:

### Step A: Registration (One-Time Setup)
1. **Record a Video:** Record an expert or trainer performing **one perfect repetition** of the exercise (e.g., a 3-second video of a perfect squat).
2. **Extract Angles:** Run our registration script `register_reference.py` on the video. The script processes each frame using MediaPipe, calculates the key joint angles, and generates a time-series matrix of shape `(N, D)` where:
   - `N` is the number of frames in the video.
   - `D` is the number of tracked angles (e.g., 3 for Squats: Hip, Knee, Ankle).
3. **Save as Binary (.npy):** Save this matrix as a lightweight NumPy binary file (e.g., `references/squat_ref.npy` which is only ~2 KB).

```
[Expert Video (.mp4)] ➔ [MediaPipe Pose] ➔ [Calculate Angles] ➔ [Save numpy array (.npy)]
```

### Step B: Real-Time Comparison (Evaluation)
1. When a user starts a rep, the system records their angle sequence frame-by-frame.
2. Once the rep ends, the user's sequence of shape `(M, D)` is compared to the expert's `(N, D)` using **Dynamic Time Warping (DTW)**.
3. DTW matches the patterns even if the user moves faster (`M < N`) or slower (`M > N`) than the expert, calculates the distance deviation, and outputs the final score (0–100).

---

## 2. Expanded Home Exercise Library (8 Exercises)

We will implement tracking and feedback rules for these 8 exercises to cover all main body muscles:

| Target Muscle | Exercise | Camera View | Key Joints Tracked | Coaching Rules & Focus |
|---|---|---|---|---|
| **Legs (Quads/Glutes)** | **Squat** | Front View | Hips, Knees, Ankles | Knee depth, knees caving inwards (Valgus warning). |
| **Legs (Hamstrings/Glutes)** | **Lunge (الطعن)** | Side View | Hips, Knees, Ankles | Forward knee angle (should not exceed 90°), rear knee depth. |
| **Chest (الصدر)** | **Push-up (الضغط)** | Side View | Shoulders, Elbows, Wrists, Hips | Hip sag (body alignment), push depth (elbow angle < 90°). |
| **Back (الظهر)** | **Superman (سوبرمان)** | Side View | Shoulders, Hips, Knees | Chest and thigh lift height (core engagement check). |
| **Shoulders (الكتف)** | **Dumbbell Overhead Press** | Front View | Shoulders, Elbows, Wrists | Arm extension symmetry, vertical alignment. |
| **Triceps (التراي)** | **Bench/Chair Dips** | Side View | Shoulders, Elbows, Wrists | Dip depth (elbow flexes to 90°), keeping back close to bench. |
| **Biceps (الباي)** | **Bicep Curl** | Side View | Shoulders, Elbows, Wrists | Keeping elbows pinned, avoiding body sway/momentum. |
| **Abs/Core (البطن)** | **Plank (البلانك)** | Side View | Shoulders, Hips, Knees, Ankles | Straight body line (hip sag/spike warning), neck neutral. |

---

## 3. Database Schema (Supabase)

```sql
create table workout_sessions (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users not null,
    exercise_type varchar(50) not null, -- 'squat', 'push_up', etc.
    rep_count integer not null,
    average_score numeric(5,2) not null,
    completed_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table workout_reps (
    id uuid default gen_random_uuid() primary key,
    session_id uuid references workout_sessions(id) on delete cascade not null,
    rep_number integer not null,
    form_score numeric(5,2) not null,
    feedback text,
    is_correct boolean not null
);
```

---

## 4. FastAPI & Flutter Connection

We will implement a local FastAPI server containing:
1. **`register_reference.py`**: Command-line tool to compile reference videos to `.npy` files.
2. **`main.py`**: WebSockets server that accepts landmark arrays from Flutter, updates rep states in real-time, speaks feedback, and saves results to Supabase.
