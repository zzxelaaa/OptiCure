import os
import requests
import time
import threading
from flask import Flask, render_template, jsonify, request
import pandas as pd
import logging
from datetime import datetime
import plotly.graph_objs as go
import json
import plotly
import RPi.GPIO as GPIO
import atexit
import signal
import sys
from scipy.optimize import differential_evolution

app = Flask(__name__)

nodeMCU_ip = "192.168.151.228"
variables = ["temp_val", "moist_val", "n_val", "p_val", "k_val", "ph_val"]
last_values = {}

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("urllib3").setLevel(logging.WARNING)

relay_air_pump = 20
relay_light_bulb = 21
relay_motor = 16


def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(relay_air_pump, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(relay_light_bulb, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(relay_motor, GPIO.OUT, initial=GPIO.HIGH)


setup_gpio()


def cleanup_gpio():
    GPIO.cleanup()
    logger.info("GPIO cleaned up")


atexit.register(cleanup_gpio)


def handle_exit(signum, frame):
    cleanup_gpio()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

sens_values = {
    "temp_val": None,
    "moist_val": None,
    "n_val": None,
    "p_val": None,
    "k_val": None,
    "ph_val": None,
}

fetched_values = {}
motor_sequence_started = False


def fetch_sensor_values():
    global fetched_values
    values = {}
    for variable_name in variables:
        if sens_values[variable_name] is not None:
            values[variable_name] = sens_values[variable_name]
        else:
            try:
                response = requests.get(f"http://{nodeMCU_ip}/{variable_name}")
                response.raise_for_status()
                variable = float(response.text)
                values[variable_name] = variable
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.debug(f"Error fetching {variable_name}: {e}")
                values[variable_name] = last_values.get(variable_name, None)
    last_values.update(values)

    fetched_values = values
    return values


def objective_function(x):
    temp_val, moist_val = x
    temp_target = 25.0
    moist_target = 50.0

    temp_error = (temp_val - temp_target) ** 2
    moist_error = (moist_val - moist_target) ** 2

    return temp_error + moist_error


def optimize_sensor_parameters():
    bounds = [(10, 40), (10, 70)]  # Temperature and moisture bounds
    result = differential_evolution(objective_function, bounds)
    return result.x


def apply_optimized_values(opt_values):
    temp_val, moist_val = opt_values

    if temp_val is not None:
        if temp_val > 30:
            if GPIO.input(relay_air_pump) != GPIO.LOW:
                GPIO.output(relay_air_pump, GPIO.LOW)
                logger.info("Air pump turned on (LOW)")
        else:
            if GPIO.input(relay_air_pump) != GPIO.HIGH:
                GPIO.output(relay_air_pump, GPIO.HIGH)
                logger.info("Air pump turned off (HIGH)")

    if moist_val is not None:
        if moist_val > 50:
            if GPIO.input(relay_light_bulb) != GPIO.LOW:
                GPIO.output(relay_light_bulb, GPIO.LOW)
                logger.info("Light bulb turned on (LOW)")
        else:
            if GPIO.input(relay_light_bulb) != GPIO.HIGH:
                GPIO.output(relay_light_bulb, GPIO.HIGH)
                logger.info("Light bulb turned off (HIGH)")


def motor_activation_sequence():
    try:
        global motor_sequence_started
        if not motor_sequence_started:
            motor_sequence_started = True
            logger.info("Initial motor activation sequence started")
            time.sleep(10)
            GPIO.output(relay_motor, GPIO.LOW)
            logger.debug("Motor on (LOW)")
            time.sleep(10)
            GPIO.output(relay_motor, GPIO.HIGH)
            logger.debug("Motor off (HIGH)")
            time.sleep(2000)
            GPIO.output(relay_motor, GPIO.LOW)
            logger.debug("Motor on (LOW)")
            time.sleep(10)
            GPIO.output(relay_motor, GPIO.HIGH)
            logger.debug("Motor off (HIGH)")
            logger.info("Initial motor activation sequence completed")
    except Exception as e:
        logger.error(f"Error in motor_activation_sequence: {e}")
        cleanup_gpio()
        sys.exit(1)


def fetch_and_record():
    try:
        while True:
            fetch_sensor_values()
            opt_values = optimize_sensor_parameters()
            apply_optimized_values(opt_values)
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error in fetch_and_record: {e}")
        cleanup_gpio()
        sys.exit(1)


def start_recording(duration):
    records = []
    end_time = time.time() + duration
    start_time = time.time()
    while time.time() < end_time:
        current_time = time.time()
        if (current_time - start_time) % 5 < 1:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            values = fetch_sensor_values()
            record = [
                timestamp,
                values["temp_val"],
                values["moist_val"],
                values["n_val"],
                values["p_val"],
                values["k_val"],
                values["ph_val"],
            ]
            records.append(record)
            logger.debug(f"Recorded values at {timestamp}: {record}")
        elapsed_time = current_time - start_time
        time_to_sleep = 1 - (elapsed_time % 1)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    df = pd.DataFrame(
        records,
        columns=[
            "timestamp",
            "temp_val",
            "moist_val",
            "n_val",
            "p_val",
            "k_val",
            "ph_val",
        ],
    )
    df.to_csv(
        "records.csv", mode="a", header=not os.path.exists("records.csv"), index=False
    )
    logger.info(f"Finished recording for {duration} seconds")


@app.route("/")
def home():
    return render_template("landing.html")


@app.route("/dashboard")
def dashboard():
    sensor_values = fetch_sensor_values()
    return render_template("dashboard.html", sensor_values=sensor_values)


@app.route("/get_sensor_values")
def get_sensor_values():
    return jsonify(fetched_values)


@app.route("/start_recording", methods=["POST"])
def start_recording_route():
    duration = int(request.form["duration"])
    threading.Thread(target=start_recording, args=(duration,)).start()
    return jsonify({"status": "Recording started", "duration": duration})


@app.route("/plot")
def plot():
    sensor_values = fetch_sensor_values()
    labels = ["Temperature", "Moisture", "Nitrogen", "Phosphorus", "Potassium", "pH"]
    values = [
        sensor_values.get("temp_val", 0),
        sensor_values.get("moist_val", 0),
        sensor_values.get("n_val", 0),
        sensor_values.get("p_val", 0),
        sensor_values.get("k_val", 0),
        sensor_values.get("ph_val", 0),
    ]
    colors = ["red", "blue", "yellow", "green", "violet", "orange"]

    data = [go.Bar(x=labels, y=values, marker=dict(color=colors))]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route("/latest_record")
def latest_record():
    try:
        df = pd.read_csv("records.csv")
        latest_record = df.iloc[-1]

        labels = [
            "Temperature",
            "Moisture",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "pH",
        ]
        values = [
            latest_record["temp_val"],
            latest_record["moist_val"],
            latest_record["n_val"],
            latest_record["p_val"],
            latest_record["k_val"],
            latest_record["ph_val"],
        ]
        timestamp = latest_record["timestamp"]

        title = f"{timestamp}"

        data = [
            go.Bar(
                x=labels,
                y=values,
                marker=dict(
                    color=["red", "blue", "yellow", "green", "violet", "orange"]
                ),
            )
        ]
        layout = go.Layout(
            title=title, xaxis=dict(title="Sensors"), yaxis=dict(title="Values")
        )

        graphJSON = json.dumps(
            dict(data=data, layout=layout), cls=plotly.utils.PlotlyJSONEncoder
        )
        return graphJSON

    except Exception as e:
        logger.error(f"Error fetching latest record: {e}")
        return jsonify([]), 500


@app.route("/history")
def history():
    try:
        df = pd.read_csv("records.csv")
        records = df.to_dict(orient="records")
        return jsonify(records)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify([]), 500


if __name__ == "__main__":
    threading.Thread(target=motor_activation_sequence).start()
    threading.Thread(target=fetch_and_record).start()
    app.run(debug=True, use_reloader=False, host="0.0.0.0")
