import json
import os
from enum import Enum
from pathlib import Path

from sklearn.linear_model import LinearRegression

from app.config import settings
from app.vbt.phases import analyze_phases
from app.vbt.track import track_disk


class ExerciseProfile(Enum):
    SQUAT = ("Barbell Squat", 0.17)
    BENCH = ("Bench Press", 0.2)
    DEADLIFT = ("Deadlift", 0.25)
    ROW = ("Barbell Row", 0.4)

    def __init__(self, key: str, speed: float):
        self.key = key
        self.cutoff_speed = speed

    @classmethod
    def from_key(cls, key: str) -> "ExerciseProfile":
        for member in cls:          # aquí la clase YA está construida
            if member.key == key:
                return member
        raise ValueError(f"perfil no válido: {key!r}")

    @property # property hace que se comporte como un atributo en vez de una funcion, por eso se le llama sin parentesis. Ademas no se puede hacer self.filepath = "jkhajsh" logicamente
    # Se le suele poner a datos del objeto, cosas que sean calculables aunque no tenga un atributo propio y se tenga que calcular. 
    # Cosas como acciones no suelen ser property porque es un proceso, no un dato.
    def filepath(self) -> Path:
        return Path(os.getenv("BASE_PROFILE_PATH")) / f"{self.name.lower()}.json"

    @property
    def thresholds(self) -> dict:
        file_path = Path(os.getenv("BASE_THRESHOLD_PATH")) / f"{self.name.lower()}.json"

        return json.loads(file_path.read_text())


class VBTProfile:
    def __init__(self,
                 exercise_name: str,
                 m: float,
                 b: float,
                 ):
        self.exercise_name = exercise_name
        self.speed_1rm = ExerciseProfile[exercise_name.upper()].value

        self.m = m
        self.b = b

    @classmethod
    def from_videos(cls, exercise_name: str, model_videos: list[tuple[str, float]]):
        """
        Basicamente la parte de process_videos y calculate baseline curve para crear el cls, y que __init__ solo acepte m y b
        """
        
        fastest_reps = cls.process_videos(model_videos) # Llamamos a la funcion que procesa los videos y devuelve la velocidad media mas alta de cada video

        # Ecuacion de la recta -> y = mx + b, donde b es el punto donde cruza peso 0, x es el peso, m la pendiente
        m, b = cls.calculate_baseline_curve(fastest_reps) # Llamamos a la funcion que calcula la ecuacion de la recta a partir de los pares (peso, velocidad)

        # guardar perfil
        cls.save_profile(exercise_name, m, b)

        return cls(exercise_name, m, b)

    @classmethod # Es class method porque necesitamos el argumento cls para crear una instancia de la clase y no tenemos ya una instancia. En este caso desde un fichero de perfil ya calculado previamente.
    def load_profile_from_file(cls, exercise_name: str) -> "VBTProfile":
        # Obtenemos el path del perfil de la clase
        profile_path = ExerciseProfile.from_key(exercise_name).filepath

        try:
            #abrir con json 
            data = json.loads(profile_path.read_text())
        except Exception as e:
            print(f"Error cargando el perfil para el ejercicio {exercise_name}: {e}")
            raise Exception

        return cls(data["m"], data["b"]) # Necesitamos un metodo para inciar la clase usando m y b

    @staticmethod
    def save_profile(exercise_name: str, m: float, b: float):
        profile_path = ExerciseProfile.from_key(exercise_name).filepath

        profile_path.write_text(json.dumps({"m": m, "b": b}))

    @staticmethod
    def process_videos(model_videos: list[tuple[str, float]]) -> list[dict]:
        """
        static method que combina logica externa con logica propia para obtener la velocidad media mas alta de cada video.
        """
        fastest_reps = []
        
        for video_path, weight in model_videos:
            print(f"Procesando video: {video_path} con peso: {weight}")
    
            try:
                result = track_disk(
                    Path(video_path), settings.model_weights, settings.conf_threshold, settings.device,
                    batch_size=settings.yolo_batch_size,
                ) # result es un objeto TrackResult con fps, tiempo, center_y (array por frame) y diameter (array por frame)
    
                _, phases = analyze_phases(result.t, result.center_y, result.diameter, settings.disk_diameter_m) # Height es un array con la altura normalizada de cada frame, phases es una lista de objetos Phase con info de cada fase (incluida velocidad media)
    
                phases.sort(key=lambda x: x.v_avg, reverse=True) # Ordenamos las fases por velocidad media descendente
                fastest_reps.append({"speed": phases[0].v_avg,
                                    "weight": weight}) # Agregamos la fase con la velocidad media más alta
    
                return fastest_reps
    
            except Exception:
                print("HA OCURRIDO ALGUN ERROR DURANTE EL PROCESAMIENTO DEL VIDEO")
                raise

    @staticmethod # Es static method porque le pasamos los videos procesados y no necesita ni la clase ni la instancia para hacer sus tareas.
    def calculate_baseline_curve(fastest_reps: list[dict]) -> tuple[float, float]:
        """
        Hay que completar los siguientes pasos:

        Regresion Lineal -> Input: Peso, Output: Velocidad
        Obtener Ecuacion de recta y guardar valores
        """
        # organizamos los datos en X e Y
        # Es mejor hacer 2 busquedas porque CPython tiene el append optimizado en el bucle y es mas rapido que hacer un for con 2 appends.
        # Si tuviesemos millones de datos seria mejor hacer un np.array del list append y luego coger la columna que queramos, pero para 4-5 datos no merece la pena.
        X = [[rep["weight"]] for rep in fastest_reps]  # Peso
        y = [rep["speed"] for rep in fastest_reps]     # Velocidad

        lr_model = LinearRegression(fit_intercept=True).fit(X, y)

        # Guardamos m y b
        return lr_model.coef_[0], lr_model.intercept_ # coef_ es la pendiente, hay que poner el indice porque devuelve un array en caso de que haya varias features, e intercept_ es el punto en el que cruza x=0

    def calculate_daily_1rm(self, phases: list):
        """
        Para calcular tu 1rm de hoy hay que hacer lo siguiente:

        Analizar el video primero

        Coges el peso y obtienes la velocidad a la que deberias ir.
        Calculas el ratio V_hoy/V_proyeccion.

        Tu 1rm de hoy es 1rm supuesto * ratio de velocidades
        """
        phases.sort(key=lambda x: x.v_avg, reverse=True)

        fastest_rep = {"speed": phases[0].v_avg,
                        "weight": phases[0].weight}

        v_modeled = self.m * fastest_rep["weight"] + self.b

        v_achieved = fastest_rep["speed"]

        v_ratio = v_achieved / v_modeled

        exercise_cutoff_speed = ExerciseProfile.from_key(self.exercise_name).cutoff_speed

        expected_1rm = (exercise_cutoff_speed - self.b) / self.m

        return expected_1rm * v_ratio

    def calculate_rir(self, phases: dict):
        """
        DOS FORMAS DE CALCULAR EL RIR - AMBAS BIEN SENCILLAS:

        1. Mirando en la tabla de thresholds, velocidad o intervalo de velocidad se corresponde con cierto RIR

        2. Miras la velocidad que pierdes de rep a rep
        - Coges la velocidad de tu ultima repe y la v_cutoff, las restas
        - Divides eso por la velocidad que pierdes rep a rep para obtener las repes que te faltan.

        Vamos a usar la primera porque es la mas sencilla, pero tambien estamos atados a los ejercicios correspondientes.
        Si en algun momento averiguas el cutoff speed de algun ejercicio, puedes a;adir el ejercicio con el cutoff
        Incluir;e el codigo para el metodo 2 debajo.

        Hay una tercera forma que es con el minimum velocity threshold, tiene el mismo problema que el metodo 1
        Muy especifico de ejercicio y dificil de a;adir nuevos ejercicios.
        """

        last_rep_speed = phases[-1].v_avg

        thresholds = ExerciseProfile.from_key(self.exercise_name).thresholds
        min_global = thresholds[0]["min_speed"] # Para detectar si una repeticion la mide debajo del treshold

        if last_rep_speed < min_global:
            return 0

        for t in thresholds:
            if t["min_speed"] <= last_rep_speed < t["max_speed"]:
                return t["RIR"]
        # Si no entra en los intervalos definidos ponemos RIR 10 para indicar que esta lejos del fallo, valor simbolico mas que util    
        return 10 
