#!/usr/bin/env python3
"""
Generate patient symptom cases and get MedGemma responses via Ollama API.
"""

import json
import subprocess
import sys
import time
from datetime import datetime

import requests

LOCAL_TUNNEL_PORT = 18080
OLLAMA_URL = f"http://localhost:{LOCAL_TUNNEL_PORT}/api/chat"

# 100 diverse patient symptom cases
PATIENT_CASES = [
    # Cardiovascular
    {"id": 1, "age": 45, "gender": "male", "symptoms": "chest pain radiating to left arm, shortness of breath, sweating"},
    {"id": 2, "age": 62, "gender": "male", "symptoms": "leg swelling, shortness of breath when lying down, fatigue"},
    {"id": 3, "age": 58, "gender": "female", "symptoms": "palpitations, dizziness, occasional fainting spells"},
    {"id": 4, "age": 70, "gender": "male", "symptoms": "chest tightness during exertion, relieved by rest"},
    {"id": 5, "age": 55, "gender": "female", "symptoms": "severe headache, nosebleeds, blurred vision, dizziness"},

    # Neurological
    {"id": 6, "age": 32, "gender": "female", "symptoms": "severe headache, sensitivity to light, nausea, stiff neck"},
    {"id": 7, "age": 72, "gender": "male", "symptoms": "memory problems, confusion, difficulty finding words, getting lost"},
    {"id": 8, "age": 75, "gender": "male", "symptoms": "sudden weakness on right side of body, slurred speech, facial droop"},
    {"id": 9, "age": 45, "gender": "female", "symptoms": "tremor in hands at rest, slow movement, rigid muscles, stooped posture"},
    {"id": 10, "age": 28, "gender": "female", "symptoms": "episodes of vision loss, numbness in limbs, fatigue, balance problems"},

    # Respiratory
    {"id": 11, "age": 67, "gender": "male", "symptoms": "persistent cough for 3 weeks, blood in sputum, weight loss"},
    {"id": 12, "age": 65, "gender": "male", "symptoms": "progressive shortness of breath, chronic cough, barrel chest"},
    {"id": 13, "age": 8, "gender": "male", "symptoms": "wheezing, shortness of breath, chest tightness, coughing at night"},
    {"id": 14, "age": 55, "gender": "female", "symptoms": "sudden shortness of breath, chest pain worse with breathing, leg swelling"},
    {"id": 15, "age": 40, "gender": "male", "symptoms": "fever, productive cough with yellow sputum, chest pain, fatigue"},

    # Endocrine
    {"id": 16, "age": 28, "gender": "female", "symptoms": "frequent urination, excessive thirst, fatigue, blurred vision"},
    {"id": 17, "age": 35, "gender": "female", "symptoms": "fatigue, weight gain, cold intolerance, dry skin, hair loss"},
    {"id": 18, "age": 38, "gender": "female", "symptoms": "palpitations, tremor, weight loss despite good appetite, anxiety"},
    {"id": 19, "age": 22, "gender": "female", "symptoms": "irregular periods, acne, excess facial hair, weight gain"},
    {"id": 20, "age": 48, "gender": "female", "symptoms": "hot flashes, night sweats, mood changes, irregular periods"},

    # Gastrointestinal
    {"id": 21, "age": 55, "gender": "male", "symptoms": "burning sensation in chest after eating, acid taste in mouth"},
    {"id": 22, "age": 25, "gender": "female", "symptoms": "abdominal cramps, diarrhea alternating with constipation, bloating"},
    {"id": 23, "age": 58, "gender": "male", "symptoms": "yellowing of skin and eyes, dark urine, itchy skin, fatigue"},
    {"id": 24, "age": 35, "gender": "male", "symptoms": "bloody diarrhea, abdominal pain, urgency, weight loss"},
    {"id": 25, "age": 50, "gender": "female", "symptoms": "right upper abdominal pain after fatty meals, nausea, bloating"},

    # Musculoskeletal
    {"id": 26, "age": 40, "gender": "female", "symptoms": "joint pain in hands and knees, morning stiffness, swelling"},
    {"id": 27, "age": 30, "gender": "female", "symptoms": "lower back pain radiating to right leg, numbness in foot"},
    {"id": 28, "age": 65, "gender": "male", "symptoms": "knee pain worse with activity, stiffness, grinding sensation"},
    {"id": 29, "age": 45, "gender": "female", "symptoms": "widespread muscle pain, fatigue, sleep problems, tender points"},
    {"id": 30, "age": 25, "gender": "male", "symptoms": "lower back pain and stiffness, worse in morning, improves with exercise"},

    # Urological/Nephrological
    {"id": 31, "age": 60, "gender": "male", "symptoms": "difficulty urinating, weak urine stream, frequent nighttime urination"},
    {"id": 32, "age": 35, "gender": "female", "symptoms": "burning urination, frequent urination, lower abdominal pain"},
    {"id": 33, "age": 45, "gender": "male", "symptoms": "severe flank pain, blood in urine, nausea, pain comes in waves"},
    {"id": 34, "age": 55, "gender": "male", "symptoms": "foamy urine, leg swelling, fatigue, decreased appetite"},
    {"id": 35, "age": 68, "gender": "female", "symptoms": "urinary incontinence, urgency, frequency, nocturia"},

    # Dermatological
    {"id": 36, "age": 33, "gender": "female", "symptoms": "skin rash on cheeks, joint pain, fatigue, mouth ulcers, sun sensitivity"},
    {"id": 37, "age": 30, "gender": "male", "symptoms": "scaly red patches on elbows and knees, nail changes, itching"},
    {"id": 38, "age": 25, "gender": "female", "symptoms": "itchy red rash in elbow creases and behind knees, dry skin"},
    {"id": 39, "age": 55, "gender": "male", "symptoms": "new mole that changed color, irregular borders, asymmetric"},
    {"id": 40, "age": 70, "gender": "female", "symptoms": "painful blistering rash on one side of torso, burning sensation"},

    # Infectious Diseases
    {"id": 41, "age": 30, "gender": "male", "symptoms": "fever, sore throat, swollen lymph nodes, fatigue, rash"},
    {"id": 42, "age": 25, "gender": "female", "symptoms": "high fever, severe headache, muscle pain, rash, joint pain"},
    {"id": 43, "age": 5, "gender": "male", "symptoms": "fever, runny nose, cough, red eyes, rash starting on face"},
    {"id": 44, "age": 35, "gender": "male", "symptoms": "night sweats, persistent cough, weight loss, fatigue, fever"},
    {"id": 45, "age": 28, "gender": "female", "symptoms": "vaginal discharge, pelvic pain, pain during intercourse, fever"},

    # Hematological/Oncological
    {"id": 46, "age": 60, "gender": "male", "symptoms": "fatigue, easy bruising, frequent infections, bone pain"},
    {"id": 47, "age": 25, "gender": "female", "symptoms": "heavy menstrual periods, fatigue, pale skin, shortness of breath"},
    {"id": 48, "age": 55, "gender": "male", "symptoms": "painless swollen lymph nodes, night sweats, weight loss, fatigue"},
    {"id": 49, "age": 70, "gender": "male", "symptoms": "bone pain, fatigue, frequent infections, increased thirst"},
    {"id": 50, "age": 45, "gender": "female", "symptoms": "breast lump, skin dimpling, nipple discharge, axillary swelling"},

    # Psychiatric/Mental Health
    {"id": 51, "age": 22, "gender": "female", "symptoms": "persistent sadness, loss of interest, sleep problems, poor concentration"},
    {"id": 52, "age": 30, "gender": "male", "symptoms": "excessive worry, restlessness, muscle tension, difficulty sleeping"},
    {"id": 53, "age": 25, "gender": "female", "symptoms": "episodes of intense fear, rapid heartbeat, sweating, feeling of doom"},
    {"id": 54, "age": 20, "gender": "male", "symptoms": "hearing voices, paranoid thoughts, social withdrawal, disorganized speech"},
    {"id": 55, "age": 35, "gender": "female", "symptoms": "mood swings, periods of high energy, impulsive behavior, decreased sleep need"},

    # ENT
    {"id": 56, "age": 42, "gender": "female", "symptoms": "recurring sinus infections, facial pain, postnasal drip, loss of smell"},
    {"id": 57, "age": 50, "gender": "male", "symptoms": "hearing loss, ringing in ears, dizziness, ear fullness"},
    {"id": 58, "age": 8, "gender": "female", "symptoms": "ear pain, fever, irritability, tugging at ear, trouble sleeping"},
    {"id": 59, "age": 55, "gender": "male", "symptoms": "persistent hoarseness, throat discomfort, cough, difficulty swallowing"},
    {"id": 60, "age": 35, "gender": "female", "symptoms": "room spinning sensation, nausea, triggered by head movements"},

    # Ophthalmological
    {"id": 61, "age": 65, "gender": "female", "symptoms": "gradual vision loss, halos around lights, eye pain, headache"},
    {"id": 62, "age": 70, "gender": "male", "symptoms": "cloudy vision, faded colors, difficulty with night driving"},
    {"id": 63, "age": 60, "gender": "female", "symptoms": "central vision loss, distorted vision, difficulty reading"},
    {"id": 64, "age": 45, "gender": "male", "symptoms": "sudden vision loss in one eye, floaters, flashing lights"},
    {"id": 65, "age": 55, "gender": "female", "symptoms": "dry eyes, burning sensation, foreign body feeling, blurred vision"},

    # Vascular
    {"id": 66, "age": 70, "gender": "male", "symptoms": "leg pain when walking, relieved by rest, cold feet, weak pulses"},
    {"id": 67, "age": 65, "gender": "female", "symptoms": "leg swelling, skin discoloration, varicose veins, leg heaviness"},
    {"id": 68, "age": 50, "gender": "male", "symptoms": "sudden severe calf pain, leg swelling, warmth, redness"},
    {"id": 69, "age": 75, "gender": "male", "symptoms": "pulsating abdominal mass, back pain, feeling of fullness"},
    {"id": 70, "age": 60, "gender": "female", "symptoms": "cold fingers turning white then blue, numbness, triggered by cold"},

    # Pediatric
    {"id": 71, "age": 2, "gender": "male", "symptoms": "high fever, barking cough, stridor, difficulty breathing at night"},
    {"id": 72, "age": 6, "gender": "female", "symptoms": "sore throat, fever, difficulty swallowing, swollen tonsils with white patches"},
    {"id": 73, "age": 4, "gender": "male", "symptoms": "fever, ear pain, vomiting, diarrhea, decreased appetite"},
    {"id": 74, "age": 10, "gender": "female", "symptoms": "excessive thirst, frequent urination, weight loss, fatigue, bedwetting"},
    {"id": 75, "age": 7, "gender": "male", "symptoms": "difficulty concentrating, hyperactivity, impulsive behavior, fidgeting"},

    # Geriatric
    {"id": 76, "age": 80, "gender": "female", "symptoms": "falls, confusion, urinary incontinence, weakness"},
    {"id": 77, "age": 85, "gender": "male", "symptoms": "decreased appetite, weight loss, fatigue, social withdrawal"},
    {"id": 78, "age": 78, "gender": "female", "symptoms": "hip pain after fall, shortened leg, external rotation"},
    {"id": 79, "age": 82, "gender": "male", "symptoms": "difficulty swallowing, coughing when eating, weight loss"},
    {"id": 80, "age": 75, "gender": "female", "symptoms": "unsteady gait, fear of falling, dizziness when standing"},

    # Autoimmune
    {"id": 81, "age": 35, "gender": "female", "symptoms": "dry eyes, dry mouth, fatigue, joint pain"},
    {"id": 82, "age": 40, "gender": "female", "symptoms": "muscle weakness, difficulty swallowing, skin rash on face and hands"},
    {"id": 83, "age": 30, "gender": "male", "symptoms": "mouth ulcers, genital ulcers, eye inflammation, skin lesions"},
    {"id": 84, "age": 45, "gender": "female", "symptoms": "skin thickening, Raynaud phenomenon, difficulty swallowing, shortness of breath"},
    {"id": 85, "age": 50, "gender": "male", "symptoms": "progressive muscle weakness, difficulty climbing stairs, muscle pain"},

    # Allergic/Immunological
    {"id": 86, "age": 28, "gender": "female", "symptoms": "sneezing, runny nose, itchy eyes, symptoms worse in spring"},
    {"id": 87, "age": 35, "gender": "male", "symptoms": "hives, swelling of lips and throat, difficulty breathing after eating peanuts"},
    {"id": 88, "age": 20, "gender": "female", "symptoms": "recurrent sinus infections, pneumonia, chronic diarrhea"},
    {"id": 89, "age": 5, "gender": "male", "symptoms": "eczema, food allergies, wheezing, family history of allergies"},
    {"id": 90, "age": 40, "gender": "female", "symptoms": "chronic fatigue, recurrent infections, joint pain, rashes"},

    # Metabolic
    {"id": 91, "age": 50, "gender": "male", "symptoms": "severe big toe pain, redness, swelling, woke up at night"},
    {"id": 92, "age": 45, "gender": "female", "symptoms": "bone pain, muscle weakness, fatigue, frequent fractures"},
    {"id": 93, "age": 35, "gender": "male", "symptoms": "abdominal pain, constipation, kidney stones, confusion"},
    {"id": 94, "age": 60, "gender": "female", "symptoms": "muscle cramps, numbness around mouth, tingling in fingers"},
    {"id": 95, "age": 55, "gender": "male", "symptoms": "bronze skin color, fatigue, joint pain, abdominal pain, diabetes"},

    # Trauma/Emergency
    {"id": 96, "age": 25, "gender": "male", "symptoms": "ankle pain after twisting injury, swelling, difficulty bearing weight"},
    {"id": 97, "age": 30, "gender": "female", "symptoms": "severe abdominal pain after car accident, bruising, dizziness"},
    {"id": 98, "age": 40, "gender": "male", "symptoms": "headache after head injury, confusion, nausea, dilated pupil"},
    {"id": 99, "age": 22, "gender": "female", "symptoms": "wrist pain after fall on outstretched hand, swelling, deformity"},
    {"id": 100, "age": 35, "gender": "male", "symptoms": "severe burn pain, blistering, redness covering arm and chest"},
]


class TunnelManager:
    """Manage IAP tunnel with automatic reconnection."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.proc = None

    def start(self):
        """Start the IAP tunnel."""
        if self.proc:
            self.stop()

        cmd = [
            "gcloud", "compute", "start-iap-tunnel",
            "medgemma-llamacpp", "8080",
            "--project", self.project_id,
            "--zone", "us-west1-a",
            "--local-host-port", f"localhost:{LOCAL_TUNNEL_PORT}",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("  Waiting for IAP tunnel to establish...")
        time.sleep(10)
        return self.is_alive()

    def stop(self):
        """Stop the IAP tunnel."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def is_alive(self):
        """Check if tunnel process is still running."""
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def ensure_connected(self):
        """Ensure tunnel is connected, restart if needed."""
        if not self.is_alive():
            print("  Tunnel died, restarting...")
            return self.start()
        return True


def query_medgemma(patient: dict, tunnel: TunnelManager, max_retries: int = 3) -> dict:
    """Send patient symptoms to MedGemma and get response with retry logic."""
    prompt = f"""Patient Information:
- Age: {patient['age']} years old
- Gender: {patient['gender']}
- Symptoms: {patient['symptoms']}

Based on these symptoms, what are the possible differential diagnoses? Please provide your clinical reasoning."""

    for attempt in range(max_retries):
        # Ensure tunnel is connected before each attempt
        if not tunnel.ensure_connected():
            print(f"    Retry {attempt + 1}/{max_retries}: Failed to establish tunnel")
            time.sleep(5)
            continue

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": "medgemma",
                    "messages": [
                        {"role": "system", "content": "You are a helpful medical assistant. Provide differential diagnoses based on patient symptoms."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                },
                timeout=300,
            )

            if response.status_code == 200:
                return {
                    "success": True,
                    "response": response.json()["message"]["content"],
                }
            else:
                print(f"    Retry {attempt + 1}/{max_retries}: HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"    Retry {attempt + 1}/{max_retries}: Request timed out")
        except requests.exceptions.ConnectionError as e:
            print(f"    Retry {attempt + 1}/{max_retries}: Connection error")
            # Force tunnel restart on connection error
            tunnel.stop()
            time.sleep(3)

        if attempt < max_retries - 1:
            time.sleep(5)

    return {"success": False, "error": f"Failed after {max_retries} attempts"}


def load_existing_results(filepath: str) -> dict:
    """Load existing results if available."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    project_id = "chncaesar1"
    output_file = "data/patient_cases_results.json"

    print("=" * 60)
    print("MedGemma Patient Case Generator")
    print("=" * 60)

    # Check for existing results to resume
    existing = load_existing_results(output_file)
    completed_ids = set()

    if existing:
        for case in existing.get("cases", []):
            if case.get("success"):
                completed_ids.add(case["patient"]["id"])
        print(f"Found {len(completed_ids)} completed cases, resuming...")

    remaining_cases = [p for p in PATIENT_CASES if p["id"] not in completed_ids][:10]  # Limit to 10 cases
    print(f"Processing {len(remaining_cases)} patient cases...")

    # Initialize tunnel manager
    tunnel = TunnelManager(project_id)
    print("\nStarting IAP tunnel...")
    if not tunnel.start():
        print("Failed to start tunnel!")
        sys.exit(1)

    # Initialize or use existing results
    if existing:
        results = existing
    else:
        results = {
            "generated_at": datetime.now().isoformat(),
            "model": "medgemma",
            "total_cases": len(PATIENT_CASES),
            "cases": []
        }

    try:
        for i, patient in enumerate(remaining_cases):
            print(f"\n[{len(completed_ids) + i + 1}/{len(PATIENT_CASES)}] Processing patient {patient['id']}...")
            print(f"  Symptoms: {patient['symptoms'][:50]}...")

            response = query_medgemma(patient, tunnel)

            case_result = {
                "patient": patient,
                "medgemma_response": response.get("response", ""),
                "success": response["success"],
            }
            if not response["success"]:
                case_result["error"] = response.get("error", "Unknown error")
                print(f"  ✗ Error: {response.get('error', 'Unknown')}")
            else:
                print(f"  ✓ Response received ({len(response['response'])} chars)")

            results["cases"].append(case_result)

            # Save checkpoint every 5 cases
            if (i + 1) % 5 == 0:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  [Checkpoint saved]")

            # Small delay between requests
            time.sleep(2)

        # Save final results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        successful = sum(1 for c in results["cases"] if c["success"])
        print("\n" + "=" * 60)
        print(f"Complete! {successful}/{len(PATIENT_CASES)} cases processed successfully")
        print(f"Results saved to: {output_file}")
        print("=" * 60)

    finally:
        print("\nClosing IAP tunnel...")
        tunnel.stop()


if __name__ == "__main__":
    main()
