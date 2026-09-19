import json
import os
import traceback
from enum import Enum
from io import BytesIO
from pathlib import Path

from sklearn.linear_model import LinearRegression

from app.config import settings
from app.vbt.phases import CONCENTRIC, analyze_phases
from app.vbt.track import track_disk


class ExerciseProfile(str, Enum):
    SQUAT = ("Barbell Squat", 0.17)
    BENCH = ("Bench Press", 0.2)
    DEADLIFT = ("Deadlift", 0.25)
    ROW = ("Barbell Row", 0.4)

    # El valor del miembro es la key (no la tupla) para que FastAPI/pydantic
    # puedan validar directamente el <select> del formulario contra el enum.
    def __new__(cls, key: str, speed: float) -> "ExerciseProfile":
        member = str.__new__(cls, key)
        member._value_ = key
        member.key = key
        member.cutoff_speed = speed
        return member

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
                 points: list[dict] | None = None,
                 ):
        self.exercise_name = exercise_name
        self.speed_1rm = ExerciseProfile.from_key(exercise_name).cutoff_speed

        self.m = m
        self.b = b
        # pares (peso, velocidad) con los que se ajusto la recta: no hacen
        # falta para calcular, pero sin ellos un perfil malo es indistinguible
        # de uno bueno al mirarlo
        self.points = points or []

    @property
    def estimated_1rm(self) -> float:
        """Peso al que la recta del perfil corta la velocidad de cutoff.

        Es el 1RM "de manual" del perfil, sin corregir por la velocidad del
        dia (para eso esta calculate_daily_1rm).
        """
        if not self.m:  # perfil degenerado: recta plana, nunca cruza el cutoff
            return float("nan")

        return (self.speed_1rm - self.b) / self.m

    @classmethod
    def from_videos(cls, exercise_name: str, model_videos: list[tuple[str, float]]):
        """
        Basicamente la parte de process_videos y calculate baseline curve para crear el cls, y que __init__ solo acepte m y b
        """
        
        fastest_reps = cls.process_videos(model_videos) # Llamamos a la funcion que procesa los videos y devuelve la velocidad media mas alta de cada video

        # Ecuacion de la recta -> y = mx + b, donde b es el punto donde cruza peso 0, x es el peso, m la pendiente
        m, b = cls.calculate_baseline_curve(fastest_reps) # Llamamos a la funcion que calcula la ecuacion de la recta a partir de los pares (peso, velocidad)

        # guardar perfil
        cls.save_profile(exercise_name, m, b, fastest_reps)

        return cls(exercise_name, m, b, fastest_reps)

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

        # points es opcional: los perfiles guardados antes no lo tienen
        return cls(exercise_name, data["m"], data["b"], data.get("points")) # Necesitamos un metodo para inciar la clase usando m y b

    @staticmethod
    def save_profile(exercise_name: str, m: float, b: float, points: list[dict] | None = None):
        profile_path = ExerciseProfile.from_key(exercise_name).filepath

        profile_path.write_text(json.dumps({"m": m, "b": b, "points": points or []}))

    def plot_profile(self) -> bytes:
        """PNG de la recta carga-velocidad: cutoff del ejercicio y 1RM.

        Devuelve los bytes en vez de escribir a disco porque el endpoint la
        sirve incrustada en el modal; el perfil guardado sigue siendo solo m y b.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        one_rm = self.estimated_1rm
        # con pendiente >= 0 la recta se aleja del cutoff y el corte no existe
        # (o sale negativo): dibujamos la recta igual, pero sin marcar el 1RM
        has_1rm = one_rm > 0
        heaviest = max((p["weight"] for p in self.points), default=0.0)
        x_max = max(one_rm * 1.05 if has_1rm else 0.0, heaviest * 1.1, 1.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        # la recta es lineal, con los dos extremos sobra
        ax.plot([0.0, x_max], [self.b, self.m * x_max + self.b],
                lw=2, color="#1f77b4", label=f"v = {self.m:.4f}·kg + {self.b:.3f}")
        ax.axhline(self.speed_1rm, ls="--", lw=1.2, color="#d62728",
                   label=f"cutoff speed — {self.speed_1rm} m/s")
        if self.points:
            ax.plot([p["weight"] for p in self.points], [p["speed"] for p in self.points],
                    "o", ms=7, color="#1f77b4", label="measured sets")
        if has_1rm:
            ax.plot([one_rm], [self.speed_1rm], "o", ms=8, color="#d62728",
                    label=f"1RM ≈ {one_rm:.1f} kg")

        ax.set_xlabel("weight (kg)")
        ax.set_ylabel("mean concentric velocity (m/s)")
        ax.set_title(f"{self.exercise_name} — velocity profile")
        ax.set_xlim(0, x_max)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)

        return buf.getvalue()

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
    
                # SOLO concentricas: en una serie pesada la bajada es mas rapida
                # que la subida, asi que coger el maximo de todas las fases mete
                # una excentrica en la recta y aplana el perfil
                concentric = [p for p in phases if p.kind == CONCENTRIC]
                if not concentric:
                    raise ValueError(f"no se han detectado repeticiones en {video_path}")

                fastest_reps.append({"speed": max(p.v_avg for p in concentric),
                                    "weight": weight}) # Agregamos la concentrica con la velocidad media más alta
    
            except Exception:
                print(f"HA OCURRIDO ALGUN ERROR DURANTE EL PROCESAMIENTO DEL VIDEO {video_path}")
                traceback.print_exc()
                raise

        return fastest_reps

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

    def calculate_daily_1rm(self, phases: list, weight: float):
        """
        Para calcular tu 1rm de hoy hay que hacer lo siguiente:

        Analizar el video primero

        Coges el peso y obtienes la velocidad a la que deberias ir.
        Calculas el ratio V_hoy/V_proyeccion.

        Tu 1rm de hoy es 1rm supuesto * ratio de velocidades
        """
        # solo concentricas, y filtrando aqui dentro: la recta se ajusto con
        # concentricas, asi que compararla contra una excentrica (mas rapida en
        # series pesadas) daria un 1RM inflado aunque el caller se despiste
        concentric = [p for p in phases if p.kind == CONCENTRIC]
        if not concentric:
            raise ValueError("no hay fases concentricas para calcular el 1RM del dia")

        # max en vez de sort para no reordenar la lista del caller
        fastest_rep = {"speed": max(p.v_avg for p in concentric),
                        "weight": weight}

        v_modeled = self.m * fastest_rep["weight"] + self.b

        v_achieved = fastest_rep["speed"]

        v_ratio = v_achieved / v_modeled

        return self.estimated_1rm * v_ratio

    def calculate_rir(self, phases: list):
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

        # la ultima repeticion es la ultima CONCENTRICA: los umbrales de RIR
        # estan medidos sobre velocidad de subida
        concentric = [p for p in phases if p.kind == CONCENTRIC]
        if not concentric:
            raise ValueError("no hay fases concentricas para estimar el RIR")

        last_rep_speed = concentric[-1].v_avg

        thresholds = ExerciseProfile.from_key(self.exercise_name).thresholds
        min_global = thresholds[0]["min_speed"] # Para detectar si una repeticion la mide debajo del treshold

        if last_rep_speed < min_global:
            return 0

        for t in thresholds:
            if t["min_speed"] <= last_rep_speed < t["max_speed"]:
                return t["RIR"]
        # Si no entra en los intervalos definidos ponemos RIR 10 para indicar que esta lejos del fallo, valor simbolico mas que util    
        return 10 
