import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Ladakh Shelter Thermal Comfort",
    page_icon="🏔️",
    layout="wide"
)

# -----------------------------
# Constants and default values
# -----------------------------

CITY_DATA = {
    "Leh": {
        "latitude": 34.15,
        "altitude": 3500,
        "winter_temperature": -12.0,
        "summer_temperature": 18.0,
        "solar_radiation": 650
    },
    "Kargil": {
        "latitude": 34.56,
        "altitude": 2676,
        "winter_temperature": -8.0,
        "summer_temperature": 22.0,
        "solar_radiation": 620
    },
    "Nubra Valley": {
        "latitude": 34.68,
        "altitude": 3048,
        "winter_temperature": -10.0,
        "summer_temperature": 20.0,
        "solar_radiation": 670
    },
    "Drass": {
        "latitude": 34.43,
        "altitude": 3280,
        "winter_temperature": -18.0,
        "summer_temperature": 16.0,
        "solar_radiation": 600
    }
}

MATERIALS = {
    "Mud brick": {
        "wall_u": 0.65,
        "roof_u": 0.55,
        "description": "Traditional material with moderate thermal mass"
    },
    "Stone masonry": {
        "wall_u": 1.20,
        "roof_u": 1.00,
        "description": "Durable but requires insulation for cold climates"
    },
    "Insulated brick": {
        "wall_u": 0.35,
        "roof_u": 0.28,
        "description": "Good thermal performance and practical construction"
    },
    "Rammed earth": {
        "wall_u": 0.45,
        "roof_u": 0.35,
        "description": "High thermal mass and suitable for passive design"
    }
}

WINDOW_TYPES = {
    "Single glazing": {
        "u_value": 5.8,
        "solar_gain": 0.70
    },
    "Double glazing": {
        "u_value": 2.8,
        "solar_gain": 0.55
    },
    "Triple glazing": {
        "u_value": 1.7,
        "solar_gain": 0.45
    }
}

# -----------------------------
# Helper functions
# -----------------------------

def saturation_vapor_pressure(temp_c):
    """Tetens approximation in Pa."""
    return 610.78 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def relative_humidity_from_dew_point(temp_c, dew_point_c):
    es = saturation_vapor_pressure(temp_c)
    e_dew = saturation_vapor_pressure(dew_point_c)
    return max(0, min(100, (e_dew / es) * 100))


def calculate_shelter(inputs):
    area = inputs["floor_area"]
    height = inputs["height"]
    volume = area * height

    wall_area = 2 * (
        inputs["length"] * height +
        inputs["width"] * height
    )

    roof_area = area
    window_area = area * inputs["window_to_wall_ratio"]
    door_area = 2.0
    opaque_wall_area = max(wall_area - window_area - door_area, 0)

    indoor_target = inputs["indoor_target"]
    outdoor_temp = inputs["outdoor_temp"]
    temperature_difference = indoor_target - outdoor_temp

    wall_u = inputs["wall_u"]
    roof_u = inputs["roof_u"]
    window_u = inputs["window_u"]
    door_u = 2.0

    wall_loss = wall_u * opaque_wall_area * temperature_difference
    roof_loss = roof_u * roof_area * temperature_difference
    window_loss = window_u * window_area * temperature_difference
    door_loss = door_u * door_area * temperature_difference

    # Simplified infiltration heat loss
    air_changes = inputs["air_changes"]
    air_density = 1.225
    specific_heat_air = 1005

    infiltration_loss = (
        air_density *
        specific_heat_air *
        volume *
        (air_changes / 3600) *
        temperature_difference
    )

    transmission_loss = (
        wall_loss +
        roof_loss +
        window_loss +
        door_loss
    )

    total_heat_loss = (
        transmission_loss +
        infiltration_loss
    )

    # Solar heat gain through windows
    solar_gain = (
        inputs["solar_radiation"] *
        window_area *
        inputs["window_solar_gain"] *
        inputs["solar_exposure_factor"]
    )

    # Internal heat gains
    occupant_gain = inputs["occupants"] * 100
    appliance_gain = inputs["appliance_gain"]

    total_heat_gain = solar_gain + occupant_gain + appliance_gain

    net_heating_load = max(total_heat_loss - total_heat_gain, 0)
    passive_surplus = max(total_heat_gain - total_heat_loss, 0)

    # Approximate indoor temperature without active heating
    effective_heat_capacity = 12000
    passive_temperature_rise = (
        total_heat_gain / effective_heat_capacity
    )

    estimated_passive_temp = outdoor_temp + passive_temperature_rise

    # Comfort assessment
    relative_humidity = relative_humidity_from_dew_point(
        indoor_target,
        inputs["dew_point"]
    )

    if indoor_target < 18:
        comfort_status = "Too cold"
    elif indoor_target > 27:
        comfort_status = "Too warm"
    elif relative_humidity < 30:
        comfort_status = "Dry"
    elif relative_humidity > 70:
        comfort_status = "Humid"
    else:
        comfort_status = "Thermally comfortable"

    # Approximate daily energy demand
    operating_hours = inputs["operating_hours"]
    daily_energy_kwh = (
        net_heating_load * operating_hours / 1000
    )

    return {
        "volume": volume,
        "wall_area": wall_area,
        "window_area": window_area,
        "transmission_loss": transmission_loss,
        "infiltration_loss": infiltration_loss,
        "total_heat_loss": total_heat_loss,
        "solar_gain": solar_gain,
        "occupant_gain": occupant_gain,
        "appliance_gain": appliance_gain,
        "total_heat_gain": total_heat_gain,
        "net_heating_load": net_heating_load,
        "passive_surplus": passive_surplus,
        "estimated_passive_temp": estimated_passive_temp,
        "relative_humidity": relative_humidity,
        "comfort_status": comfort_status,
        "daily_energy_kwh": daily_energy_kwh
    }


def create_hourly_profile(inputs, results):
    hours = np.arange(0, 24)

    outdoor_profile = (
        inputs["outdoor_temp"] +
        4 * np.sin((hours - 8) * np.pi / 12)
    )

    solar_profile = np.maximum(
        0,
        np.sin((hours - 6) * np.pi / 12)
    ) * inputs["solar_radiation"]

    heat_loss_profile = np.maximum(
        inputs["indoor_target"] - outdoor_profile,
        0
    ) * (
        results["total_heat_loss"] /
        max(inputs["indoor_target"] - inputs["outdoor_temp"], 1)
    )

    solar_gain_profile = (
        solar_profile *
        results["window_area"] *
        inputs["window_solar_gain"] *
        inputs["solar_exposure_factor"]
    )

    heating_load_profile = np.maximum(
        heat_loss_profile -
        solar_gain_profile -
        results["occupant_gain"] -
        results["appliance_gain"],
        0
    )

    return pd.DataFrame({
        "Hour": hours,
        "Outdoor Temperature (°C)": outdoor_profile,
        "Solar Radiation (W/m²)": solar_profile,
        "Heating Load (W)": heating_load_profile
    }).set_index("Hour")


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🏔️ Shelter Design")

city = st.sidebar.selectbox(
    "Select location",
    list(CITY_DATA.keys())
)

season = st.sidebar.selectbox(
    "Design season",
    ["Winter", "Summer"]
)

city_info = CITY_DATA[city]

if season == "Winter":
    default_outdoor_temp = city_info["winter_temperature"]
else:
    default_outdoor_temp = city_info["summer_temperature"]

st.sidebar.info(
    f"Altitude: {city_info['altitude']} m\n\n"
    f"Latitude: {city_info['latitude']}°"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This prototype provides preliminary estimates. "
    "Validate final designs using field measurements and CFD simulation."
)

# -----------------------------
# Main title
# -----------------------------

st.title("🏔️ Ladakh Shelter Thermal Comfort Model")

st.markdown(
    """
    This software estimates shelter heat loss, passive solar gain,
    heating demand, and thermal comfort for high-altitude cold-desert regions.
    """
)

# -----------------------------
# Input form
# -----------------------------

with st.form("shelter_design_form"):
    st.subheader("Shelter Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        floor_area = st.number_input(
            "Floor area (m²)",
            min_value=5.0,
            max_value=500.0,
            value=40.0,
            step=1.0
        )

        length = st.number_input(
            "Shelter length (m)",
            min_value=2.0,
            max_value=30.0,
            value=8.0,
            step=0.5
        )

        width = st.number_input(
            "Shelter width (m)",
            min_value=2.0,
            max_value=30.0,
            value=5.0,
            step=0.5
        )

        height = st.number_input(
            "Interior height (m)",
            min_value=2.0,
            max_value=8.0,
            value=3.0,
            step=0.1
        )

    with col2:
        material = st.selectbox(
            "Primary wall material",
            list(MATERIALS.keys())
        )

        insulation_thickness = st.number_input(
            "Additional insulation thickness (mm)",
            min_value=0,
            max_value=300,
            value=50,
            step=10
        )

        window_type = st.selectbox(
            "Window type",
            list(WINDOW_TYPES.keys())
        )

        window_to_wall_ratio = st.slider(
            "Window-to-wall ratio",
            min_value=0.02,
            max_value=0.50,
            value=0.15,
            step=0.01
        )

    with col3:
        occupants = st.number_input(
            "Number of occupants",
            min_value=1,
            max_value=50,
            value=4,
            step=1
        )

        indoor_target = st.number_input(
            "Target indoor temperature (°C)",
            min_value=10.0,
            max_value=35.0,
            value=21.0,
            step=0.5
        )

        outdoor_temp = st.number_input(
            "Outdoor temperature (°C)",
            min_value=-40.0,
            max_value=45.0,
            value=float(default_outdoor_temp),
            step=0.5
        )

        dew_point = st.number_input(
            "Outdoor dew point (°C)",
            min_value=-40.0,
            max_value=35.0,
            value=-18.0,
            step=0.5
        )

    st.subheader("Environmental and Operational Parameters")

    col4, col5, col6 = st.columns(3)

    with col4:
        solar_radiation = st.number_input(
            "Solar radiation (W/m²)",
            min_value=0,
            max_value=1500,
            value=int(city_info["solar_radiation"]),
            step=25
        )

        solar_exposure_factor = st.slider(
            "Solar exposure factor",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05
        )

    with col5:
        air_changes = st.number_input(
            "Air changes per hour",
            min_value=0.1,
            max_value=10.0,
            value=0.7,
            step=0.1
        )

        appliance_gain = st.number_input(
            "Appliance heat gain (W)",
            min_value=0,
            max_value=5000,
            value=300,
            step=50
        )

    with col6:
        operating_hours = st.number_input(
            "Heating operation per day (hours)",
            min_value=1.0,
            max_value=24.0,
            value=12.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Calculate Thermal Performance",
        type="primary"
    )

# -----------------------------
# Calculation
# -----------------------------

if submitted or "results" not in st.session_state:
    selected_material = MATERIALS[material]
    selected_window = WINDOW_TYPES[window_type]

    # Approximate improvement from insulation
    wall_u = selected_material["wall_u"] * (
        1 / (1 + insulation_thickness / 100)
    )

    roof_u = selected_material["roof_u"] * (
        1 / (1 + insulation_thickness / 80)
    )

    inputs = {
        "floor_area": floor_area,
        "length": length,
        "width": width,
        "height": height,
        "wall_u": wall_u,
        "roof_u": roof_u,
        "window_u": selected_window["u_value"],
        "window_solar_gain": selected_window["solar_gain"],
        "window_to_wall_ratio": window_to_wall_ratio,
        "occupants": occupants,
        "indoor_target": indoor_target,
        "outdoor_temp": outdoor_temp,
        "dew_point": dew_point,
        "solar_radiation": solar_radiation,
        "solar_exposure_factor": solar_exposure_factor,
        "air_changes": air_changes,
        "appliance_gain": appliance_gain,
        "operating_hours": operating_hours
    }

    results = calculate_shelter(inputs)

    st.session_state["results"] = results
    st.session_state["inputs"] = inputs
    st.session_state["city"] = city
    st.session_state["material"] = material
    st.session_state["window_type"] = window_type

else:
    results = st.session_state["results"]
    inputs = st.session_state["inputs"]

# -----------------------------
# Results dashboard
# -----------------------------

st.markdown("---")
st.subheader("Thermal Performance Results")

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Total heat loss",
    f"{results['total_heat_loss']:.0f} W"
)

r2.metric(
    "Net heating load",
    f"{results['net_heating_load']:.0f} W"
)

r3.metric(
    "Daily energy demand",
    f"{results['daily_energy_kwh']:.1f} kWh"
)

r4.metric(
    "Relative humidity",
    f"{results['relative_humidity']:.1f}%"
)

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Comfort Assessment")

    if results["comfort_status"] == "Thermally comfortable":
        st.success(results["comfort_status"])
    elif results["comfort_status"] in ["Too cold", "Too warm"]:
        st.error(results["comfort_status"])
    else:
        st.warning(results["comfort_status"])

    st.write(
        f"Estimated passive indoor temperature: "
        f"**{results['estimated_passive_temp']:.1f} °C**"
    )

    st.write(
        f"Target indoor temperature: "
        f"**{inputs['indoor_target']:.1f} °C**"
    )

    st.write(
        f"Selected location: **{st.session_state['city']}**"
    )

    st.write(
        f"Wall material: **{st.session_state['material']}**"
    )

    st.write(
        f"Window type: **{st.session_state['window_type']}**"
    )

with right:
    st.subheader("Heat Balance")

    heat_balance = pd.DataFrame({
        "Heat Component": [
            "Wall, roof, window and door loss",
            "Infiltration loss",
            "Solar gain",
            "Occupant gain",
            "Appliance gain"
        ],
        "Value (W)": [
            results["transmission_loss"],
            results["infiltration_loss"],
            results["solar_gain"],
            results["occupant_gain"],
            results["appliance_gain"]
        ]
    })

    st.bar_chart(
        heat_balance.set_index("Heat Component")
    )

# -----------------------------
# Hourly profile
# -----------------------------

st.markdown("---")
st.subheader("24-Hour Thermal Profile")

profile = create_hourly_profile(inputs, results)

st.line_chart(
    profile[
        [
            "Outdoor Temperature (°C)",
            "Heating Load (W)"
        ]
    ]
)

with st.expander("View hourly data"):
    st.dataframe(profile, use_container_width=True)

# -----------------------------
# Recommendations
# -----------------------------

st.markdown("---")
st.subheader("Design Recommendations")

recommendations = []

if results["net_heating_load"] > 2500:
    recommendations.append(
        "Increase wall and roof insulation or reduce air leakage."
    )

if results["infiltration_loss"] > results["transmission_loss"] * 0.35:
    recommendations.append(
        "Infiltration loss is high. Improve door sealing, window sealing, "
        "and controlled ventilation."
    )

if results["solar_gain"] < 500:
    recommendations.append(
        "Increase south-facing glazing or improve solar exposure where possible."
    )

if inputs["window_to_wall_ratio"] > 0.30:
    recommendations.append(
        "The glazing ratio is high for a cold climate. Consider better glazing "
        "or reduce window area."
    )

if results["relative_humidity"] < 30:
    recommendations.append(
        "Indoor air may become dry. Consider controlled humidification."
    )

if not recommendations:
    recommendations.append(
        "The selected design has a reasonable preliminary thermal balance."
    )

for recommendation in recommendations:
    st.write(f"• {recommendation}")

# -----------------------------
# Export report
# -----------------------------

st.markdown("---")
st.subheader("Download Results")

export_data = pd.DataFrame({
    "Parameter": [
        "Location",
        "Season",
        "Floor area (m²)",
        "Outdoor temperature (°C)",
        "Target indoor temperature (°C)",
        "Total heat loss (W)",
        "Net heating load (W)",
        "Daily energy demand (kWh)",
        "Solar gain (W)",
        "Infiltration loss (W)",
        "Relative humidity (%)",
        "Comfort status"
    ],
    "Value": [
        st.session_state["city"],
        season,
        inputs["floor_area"],
        inputs["outdoor_temp"],
        inputs["indoor_target"],
        round(results["total_heat_loss"], 2),
        round(results["net_heating_load"], 2),
        round(results["daily_energy_kwh"], 2),
        round(results["solar_gain"], 2),
        round(results["infiltration_loss"], 2),
        round(results["relative_humidity"], 2),
        results["comfort_status"]
    ]
})

csv_data = export_data.to_csv(index=False)

st.download_button(
    label="Download thermal report as CSV",
    data=csv_data,
    file_name="ladakh_shelter_thermal_report.csv",
    mime="text/csv"
)

st.caption(
    "Prototype note: This model uses simplified heat-balance equations. "
    "For final engineering validation, connect it with measured weather data "
    "and ANSYS Fluent or another CFD solver."
)