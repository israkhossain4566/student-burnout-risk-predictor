"""Preset scenarios for the burnout risk demo."""

SCENARIOS = {
    "Healthy Routine": {
        "sleep_hours": 8.0,
        "stress_level": "Low",
        "physical_activity_min": 60,
        "academic_workload": "Low",
        "social_interaction": "High",
        "description": "A student with a balanced schedule, regular exercise, and strong social support."
    },
    "Moderate Academic Pressure": {
        "sleep_hours": 6.5,
        "stress_level": "Medium",
        "physical_activity_min": 30,
        "academic_workload": "Medium",
        "social_interaction": "Medium",
        "description": "Typical mid-semester conditions with manageable but persistent assignments."
    },
    "Exam Week Overload": {
        "sleep_hours": 4.5,
        "stress_level": "High",
        "physical_activity_min": 10,
        "academic_workload": "High",
        "social_interaction": "Low",
        "description": "Severe lack of sleep, restricted social life, and high academic pressure during exams."
    },
    "Chronic Burnout Pattern": {
        "sleep_hours": 4.0,
        "stress_level": "High",
        "physical_activity_min": 0,
        "academic_workload": "High",
        "social_interaction": "Low",
        "description": "Long-term high stress and isolation, characteristic of severe burnout-risk."
    }
}
