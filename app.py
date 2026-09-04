import streamlit as st
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fuzzy Traffic Congestion Evaluation",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Fuzzy Logic-Based Traffic Congestion Evaluation System")

st.write(
    "This AI-based system evaluates traffic congestion using "
    "Vehicle Density, Average Speed, and Average Waiting Time."
)

st.info(
    "AI Technique Used: Mamdani Fuzzy Logic Inference System"
)


# ============================================================
# FUZZY UNIVERSES
# ============================================================

density = np.arange(0, 101, 1)
speed = np.arange(0, 61, 1)
waiting = np.arange(0, 121, 1)
congestion = np.arange(0, 101, 1)


# ============================================================
# MEMBERSHIP FUNCTIONS
# ============================================================

# Vehicle Density
density_low = fuzz.trimf(density, [0, 20, 40])
density_medium = fuzz.trimf(density, [20, 45, 70])
density_high = fuzz.trapmf(density, [50, 75, 100, 100])


# Average Speed
speed_low = fuzz.trapmf(speed, [0, 0, 15, 30])
speed_medium = fuzz.trimf(speed, [15, 30, 45])
speed_high = fuzz.trapmf(speed, [35, 50, 60, 60])


# Waiting Time
waiting_short = fuzz.trapmf(waiting, [0, 0, 20, 40])
waiting_moderate = fuzz.trimf(waiting, [20, 50, 80])
waiting_long = fuzz.trapmf(waiting, [60, 90, 120, 120])


# Output: Congestion
congestion_low = fuzz.trapmf(congestion, [0, 0, 15, 30])
congestion_moderate = fuzz.trimf(congestion, [20, 40, 60])
congestion_high = fuzz.trimf(congestion, [50, 65, 80])
congestion_severe = fuzz.trapmf(congestion, [70, 85, 100, 100])


# ============================================================
# FUZZY EVALUATION FUNCTION
# ============================================================

def evaluate_congestion(vehicle_density, average_speed, waiting_time):

    # --------------------------------------------------------
    # FUZZIFICATION
    # --------------------------------------------------------

    density_low_level = fuzz.interp_membership(
        density, density_low, vehicle_density
    )

    density_medium_level = fuzz.interp_membership(
        density, density_medium, vehicle_density
    )

    density_high_level = fuzz.interp_membership(
        density, density_high, vehicle_density
    )


    speed_low_level = fuzz.interp_membership(
        speed, speed_low, average_speed
    )

    speed_medium_level = fuzz.interp_membership(
        speed, speed_medium, average_speed
    )

    speed_high_level = fuzz.interp_membership(
        speed, speed_high, average_speed
    )


    waiting_short_level = fuzz.interp_membership(
        waiting, waiting_short, waiting_time
    )

    waiting_moderate_level = fuzz.interp_membership(
        waiting, waiting_moderate, waiting_time
    )

    waiting_long_level = fuzz.interp_membership(
        waiting, waiting_long, waiting_time
    )


    # --------------------------------------------------------
    # FUZZY RULES
    # --------------------------------------------------------

    rules = []

    # LOW CONGESTION

    rule1 = min(
        density_low_level,
        speed_high_level,
        waiting_short_level
    )

    rule2 = min(
        density_low_level,
        speed_medium_level,
        waiting_short_level
    )

    rules.append(("Low", rule1, congestion_low))
    rules.append(("Low", rule2, congestion_low))


    # MODERATE CONGESTION

    rule3 = min(
        density_low_level,
        speed_low_level,
        waiting_short_level
    )

    rule4 = min(
        density_low_level,
        speed_low_level,
        waiting_long_level
    )

    rule5 = min(
        density_medium_level,
        speed_medium_level,
        waiting_moderate_level
    )

    rule6 = min(
        density_medium_level,
        speed_high_level,
        waiting_short_level
    )

    rules.append(("Moderate", rule3, congestion_moderate))
    rules.append(("Moderate", rule4, congestion_moderate))
    rules.append(("Moderate", rule5, congestion_moderate))
    rules.append(("Moderate", rule6, congestion_moderate))


    # HIGH CONGESTION

    rule7 = min(
        density_medium_level,
        speed_low_level,
        waiting_moderate_level
    )

    rule8 = min(
        density_medium_level,
        speed_low_level,
        waiting_long_level
    )

    rule9 = min(
        density_high_level,
        speed_medium_level,
        waiting_moderate_level
    )

    rule10 = min(
        density_high_level,
        speed_low_level,
        waiting_moderate_level
    )

    rule11 = min(
        density_high_level,
        speed_high_level,
        waiting_long_level
    )

    rules.append(("High", rule7, congestion_high))
    rules.append(("High", rule8, congestion_high))
    rules.append(("High", rule9, congestion_high))
    rules.append(("High", rule10, congestion_high))
    rules.append(("High", rule11, congestion_high))


    # SEVERE CONGESTION

    rule12 = min(
        density_high_level,
        speed_low_level,
        waiting_long_level
    )

    rule13 = min(
        density_high_level,
        speed_medium_level,
        waiting_long_level
    )

    rules.append(("Severe", rule12, congestion_severe))
    rules.append(("Severe", rule13, congestion_severe))


    # --------------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------------

    aggregated = np.zeros_like(congestion, dtype=float)

    for level, strength, membership in rules:

        clipped_membership = np.fmin(
            strength,
            membership
        )

        aggregated = np.fmax(
            aggregated,
            clipped_membership
        )


    # --------------------------------------------------------
    # DEFUZZIFICATION
    # --------------------------------------------------------

    if np.sum(aggregated) == 0:

        congestion_score = 0

    else:

        congestion_score = fuzz.defuzz(
            congestion,
            aggregated,
            'centroid'
        )


    # --------------------------------------------------------
    # CONGESTION LEVEL
    # --------------------------------------------------------

    if congestion_score <= 30:

        congestion_level = "LOW"

    elif congestion_score <= 60:

        congestion_level = "MODERATE"

    elif congestion_score <= 80:

        congestion_level = "HIGH"

    else:

        congestion_level = "SEVERE"


    memberships = {

        "Density - Low": density_low_level,
        "Density - Medium": density_medium_level,
        "Density - High": density_high_level,

        "Speed - Low": speed_low_level,
        "Speed - Medium": speed_medium_level,
        "Speed - High": speed_high_level,

        "Waiting - Short": waiting_short_level,
        "Waiting - Moderate": waiting_moderate_level,
        "Waiting - Long": waiting_long_level

    }


    return congestion_score, congestion_level, memberships, aggregated


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📊 Traffic Input")

col1, col2, col3 = st.columns(3)


with col1:

    vehicle_density = st.slider(
        "Vehicle Density",
        min_value=0,
        max_value=100,
        value=60,
        help="Approximate number of vehicles in the selected road/intersection area."
    )

    st.caption("Range: 0–100 vehicles")


with col2:

    average_speed = st.slider(
        "Average Speed (km/h)",
        min_value=0,
        max_value=60,
        value=30,
        help="Average speed of vehicles."
    )

    st.caption("Range: 0–60 km/h")


with col3:

    waiting_time = st.slider(
        "Average Waiting Time (seconds)",
        min_value=0,
        max_value=120,
        value=45,
        help="Average time vehicles spend waiting."
    )

    st.caption("Range: 0–120 seconds")


# ============================================================
# EVALUATION
# ============================================================

score, level, memberships, aggregated = evaluate_congestion(
    vehicle_density,
    average_speed,
    waiting_time
)


# ============================================================
# RESULT
# ============================================================

st.header("🎯 Congestion Evaluation")

result_col1, result_col2 = st.columns(2)


with result_col1:

    st.metric(
        "Congestion Score",
        f"{score:.2f} / 100"
    )


with result_col2:

    st.metric(
        "Congestion Level",
        level
    )


if level == "LOW":

    st.success(
        "🟢 Low congestion — traffic is moving relatively smoothly."
    )

elif level == "MODERATE":

    st.info(
        "🟡 Moderate congestion — traffic conditions are beginning to slow down."
    )

elif level == "HIGH":

    st.warning(
        "🟠 High congestion — significant traffic delay is present."
    )

else:

    st.error(
        "🔴 Severe congestion — major traffic delays are expected."
    )


# ============================================================
# FUZZIFICATION RESULTS
# ============================================================

st.header("🧠 Fuzzification Results")

st.write(
    "These values represent the degree of membership of each input "
    "in its fuzzy category. Values range from 0 to 1."
)


membership_col1, membership_col2, membership_col3 = st.columns(3)


with membership_col1:

    st.subheader("Vehicle Density")

    st.write(
        f"Low: **{memberships['Density - Low']:.3f}**"
    )

    st.write(
        f"Medium: **{memberships['Density - Medium']:.3f}**"
    )

    st.write(
        f"High: **{memberships['Density - High']:.3f}**"
    )


with membership_col2:

    st.subheader("Average Speed")

    st.write(
        f"Low: **{memberships['Speed - Low']:.3f}**"
    )

    st.write(
        f"Medium: **{memberships['Speed - Medium']:.3f}**"
    )

    st.write(
        f"High: **{memberships['Speed - High']:.3f}**"
    )


with membership_col3:

    st.subheader("Waiting Time")

    st.write(
        f"Short: **{memberships['Waiting - Short']:.3f}**"
    )

    st.write(
        f"Moderate: **{memberships['Waiting - Moderate']:.3f}**"
    )

    st.write(
        f"Long: **{memberships['Waiting - Long']:.3f}**"
    )


# ============================================================
# OUTPUT GRAPH
# ============================================================

st.header("📈 Fuzzy Output Analysis")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    congestion,
    congestion_low,
    label="Low"
)

ax.plot(
    congestion,
    congestion_moderate,
    label="Moderate"
)

ax.plot(
    congestion,
    congestion_high,
    label="High"
)

ax.plot(
    congestion,
    congestion_severe,
    label="Severe"
)

ax.fill_between(
    congestion,
    0,
    aggregated,
    alpha=0.35,
    label="Aggregated Output"
)

ax.axvline(
    score,
    linestyle="--",
    linewidth=2,
    label=f"Defuzzified Score = {score:.2f}"
)

ax.set_xlabel("Congestion Score")
ax.set_ylabel("Membership Degree")

ax.set_title(
    "Fuzzy Congestion Output Membership Functions"
)

ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)

ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.header("⚙️ How the AI System Works")

st.markdown(
    """
**1. Input**

The system receives:

- Vehicle Density
- Average Speed
- Average Waiting Time

**2. Fuzzification**

Numerical values are converted into fuzzy linguistic categories such as:

- Low
- Medium
- High
- Short
- Moderate
- Long

**3. Rule Evaluation**

The fuzzy rules determine how strongly each traffic condition applies.

Example:

> IF vehicle density is HIGH AND speed is LOW AND waiting time is LONG  
> THEN congestion is SEVERE.

**4. Aggregation**

The outputs of all activated fuzzy rules are combined.

**5. Defuzzification**

The aggregated fuzzy result is converted into one numerical congestion score from 0 to 100.

**6. Final Decision**

The score is classified as:

- LOW
- MODERATE
- HIGH
- SEVERE
"""
)


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander("⚠️ Project Limitations"):

    st.write(
        """
        • The current system uses manually provided traffic inputs.

        • Membership-function ranges are designed for this project and
          are not trained from a large real-world traffic dataset.

        • The system evaluates a traffic condition at a given moment
          rather than continuously monitoring a live road.

        • Real-world deployment could use traffic sensors or camera-based
          vehicle detection to automatically provide the inputs.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fuzzy Logic-Based Traffic Congestion Evaluation System | "
    "Python + scikit-fuzzy + Streamlit"
)