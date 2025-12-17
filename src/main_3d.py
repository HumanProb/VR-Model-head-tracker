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
        self.prev_yaw = 0.0

        # -------- LOAD MODEL --------
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_os_path = os.path.join(project_root, "assets", "face.obj")
        model_path = Filename.fromOsSpecific(model_os_path)
        model_path.makeTrueCase()

        # 🔑 Create a pivot node
        self.pivot = self.render.attachNewNode("pivot")

        self.model = self.loader.loadModel(model_path)
        self.model.reparentTo(self.pivot)

        # Model scale & position
        self.model.setScale(2)
        self.model.setPos(0, 0, 0)

        # 🔑 FIX MODEL ORIENTATION ONCE
        # These values correct upside-down / backward models
        self.model.setHpr(0, 90, 0)
        

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
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)
        self.taskMgr.add(self.update, "update")
        
    def get_head_direction(self, pitch, yaw):
        # LEFT / RIGHT
        if yaw < -20:
            lr = "VERY LEFT"
        elif yaw < -8:
            lr = "LEFT"
        elif yaw > 20:
            lr = "VERY RIGHT"
        elif yaw > 8:
            lr = "RIGHT"
        else:
            lr = "CENTER"

        # UP / DOWN
        if pitch < -15:
            ud = "VERY DOWN"
        elif pitch < -5:
            ud = "DOWN"
        elif pitch > 15:
            ud = "VERY UP"
        elif pitch > 5:
            ud = "UP"
        else:
            ud = "FRONT"

        return lr, ud





    def update(self, task):
        success, frame = self.cap.read()
        if not success:
            return task.cont

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            pitch, yaw, _ = estimate_head_pose(landmarks, w, h)

            # Normalize
            norm_yaw = yaw / 8.0
            norm_pitch = -pitch / 8.0

            # Clamp
            norm_yaw = max(min(norm_yaw, 30), -30)
            norm_pitch = max(min(norm_pitch, 25), -25)

            # Smooth
            self.prev_yaw = 0.8 * self.prev_yaw + 0.2 * norm_yaw
            self.prev_pitch = 0.8 * getattr(self, "prev_pitch", 0) + 0.2 * norm_pitch

            # Apply (AMPLIFIED)
            self.pivot.setH(self.prev_yaw * 6)
            self.pivot.setP(self.prev_pitch * 2)

        else:
            # Return to neutral
            self.prev_yaw *= 0.9
            self.prev_pitch *= 0.9

            self.pivot.setH(self.prev_yaw * 3)
            self.pivot.setP(self.prev_pitch * 2)    



        return task.cont


app = FaceDriven3D()
app.run()
