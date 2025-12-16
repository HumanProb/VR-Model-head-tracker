HEAD
from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, AmbientLight, Vec4, Filename
import cv2
import mediapipe as mp
import os

from .face_tracker import estimate_head_pose


class FaceDriven3D(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()

        # -------- STATE --------
        self.prev_yaw = 0.0

        # -------- LOAD MODEL (CORRECT WINDOWS FIX) --------
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_os_path = os.path.join(project_root, "assets", "face.obj")

        print("Loading model from:", model_os_path)

        model_panda_path = Filename.fromOsSpecific(model_os_path)
        model_panda_path.makeTrueCase()

        self.model = self.loader.loadModel(model_panda_path)
        self.model.reparentTo(self.render)
        self.model.setScale(2)
        self.model.setPos(0, 0, 0)

        # -------- CAMERA --------
        self.cam.setPos(0, -8, 2)
        self.cam.lookAt(0, 0, 1)

        # -------- LIGHTING --------
        dlight = DirectionalLight("dlight")
        dlight.setColor(Vec4(1, 1, 1, 1))
        dlight_np = self.render.attachNewNode(dlight)
        dlight_np.setHpr(-30, -60, 0)
        self.render.setLight(dlight_np)

        alight = AmbientLight("alight")
        alight.setColor(Vec4(0.4, 0.4, 0.4, 1))
        alight_np = self.render.attachNewNode(alight)
        self.render.setLight(alight_np)

        # -------- MEDIAPIPE --------
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

        self.cap = cv2.VideoCapture(0)
        self.taskMgr.add(self.update, "update")

    def update(self, task):
        success, frame = self.cap.read()
        if not success:
            return task.cont

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("NO FACE DETECTED")
            return task.cont

        print("FACE DETECTED")

        landmarks = results.multi_face_landmarks[0].landmark
        _, yaw, _ = estimate_head_pose(landmarks, w, h)

        print("Raw yaw:", yaw)


app = FaceDriven3D()
app.run()

from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, AmbientLight, Vec4, Filename
import cv2
import mediapipe as mp
import os

from .face_tracker import estimate_head_pose


class FaceDriven3D(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()

        # -------- STATE --------
        self.prev_yaw = 0.0

        # -------- LOAD MODEL (CORRECT WINDOWS FIX) --------
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_os_path = os.path.join(project_root, "assets", "face.obj")

        print("Loading model from:", model_os_path)

        model_panda_path = Filename.fromOsSpecific(model_os_path)
        model_panda_path.makeTrueCase()

        self.model = self.loader.loadModel(model_panda_path)
        self.model.reparentTo(self.render)
        self.model.setScale(2)
        self.model.setPos(0, 2, 0)
        self.model.setP(30) 

        # -------- CAMERA --------
        self.cam.setPos(0, -8, 2)
        self.cam.lookAt(0, 0, 1)

        # -------- LIGHTING --------
        dlight = DirectionalLight("dlight")
        dlight.setColor(Vec4(1, 1, 1, 1))
        dlight_np = self.render.attachNewNode(dlight)
        dlight_np.setHpr(-30, -60, 0)
        self.render.setLight(dlight_np)

        alight = AmbientLight("alight")
        alight.setColor(Vec4(0.4, 0.4, 0.4, 1))
        alight_np = self.render.attachNewNode(alight)
        self.render.setLight(alight_np)

        # -------- MEDIAPIPE --------
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

        self.cap = cv2.VideoCapture(0)
        self.taskMgr.add(self.update, "update")

    def update(self, task):
        success, frame = self.cap.read()
        if not success:
            return task.cont

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("NO FACE DETECTED")
            return task.cont

        print("FACE DETECTED")

        landmarks = results.multi_face_landmarks[0].landmark
        _, yaw, _ = estimate_head_pose(landmarks, w, h)

        print("Raw yaw:", yaw)


app = FaceDriven3D()
app.run()
