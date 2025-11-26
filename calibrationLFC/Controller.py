from InitialCalibration import InitialCalibration
from ResultLogger import ResultLogger

class Controller:

    def __init__(self):
        self.initialCalibration = InitialCalibration()
        self.scoreHistory = []
        self.currentState = None
        self.resultLogger = ResultLogger()

    def runInitialCalibration(self, imageSet):
        print("Controller: starting initial calibration...")
        calibrationState = self.initialCalibration.run(imageSet)

        meta = {
            "runType": "initial",
            "numPoses": imageSet.numPoses,
            "imageDir": imageSet.baseDir,
            "cameraIds": imageSet.cameraIds,
        }
        self.resultLogger.logInitialCalibration(calibrationState, meta=meta)

        return calibrationState
    
    def recalibration(self, imageSet):
        pass

    def selfHealthCheck(self, imageSet):
        pass
