import cv2
import numpy as np
import time

def identify_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "Triangulo"
    elif sides == 4:
        return "Retangulo/Quadrado"
    elif sides == 5:
        return "Pentagono"
    elif sides == 6:
      return "Hexagono"
    elif sides == 10:
      return "Estrela"
    elif sides == 12:
      return "Cruz"
    elif sides > 12:
        return "Circulo"
    return "Desconhecido"


def setLimitsOfTrackbar():
    hue = {}
    hue["min"] = cv2.getTrackbarPos("Min Hue", trackbarWindow)
    hue["max"] = cv2.getTrackbarPos("Max Hue", trackbarWindow)
    
    if hue["min"] > hue["max"]:
        cv2.setTrackbarPos("Max Hue", trackbarWindow, hue["min"])
        hue["max"] = cv2.getTrackbarPos("Max Hue", trackbarWindow)
    
    sat = {}
    sat["min"] = cv2.getTrackbarPos("Min Saturation", trackbarWindow)
    sat["max"] = cv2.getTrackbarPos("Max Saturation", trackbarWindow)
    
    if sat["min"] > sat["max"]:
        cv2.setTrackbarPos("Max Saturation", trackbarWindow, sat["min"])
        sat["max"] = cv2.getTrackbarPos("Max Saturation", trackbarWindow)

    val = {}
    val["min"] = cv2.getTrackbarPos("Min Value", trackbarWindow)
    val["max"] = cv2.getTrackbarPos("Max Value", trackbarWindow)
    
    if val["min"] > val["max"]:
        cv2.setTrackbarPos("Max Value", trackbarWindow, val["min"])
        val["max"] = cv2.getTrackbarPos("Max Value", trackbarWindow)
        
    return hue, sat, val

def computeTracking(frame, hue, sat, val):
    
    #transforma a imagem de RGB para HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #definir os intervalos de cores que vão aparecer na imagem final
    lowerColor = np.array([hue['min'], sat["min"], val["min"]])
    upperColor = np.array([hue['max'], sat["max"], val["max"]])
    
    #marcador pra saber se o pixel pertence ao intervalo ou não
    mask = cv2.inRange(hsvImage, lowerColor, upperColor)
    
    #aplica máscara que "deixa passar" pixels pertencentes ao intervalo, como filtro
    result = cv2.bitwise_and(frame, frame, mask = mask)
    
    #aplica limiarização
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #encontra pontos que circundam regiões conexas (contour)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos pequenos
    min_area = 500  # Ajuste conforme necessário
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    #se existir contornos 
    maxArea = 0.0   
    cntMaxArea = None
    if contours:
        #retornando a área do primeiro grupo de pixels brancos
        maxArea = cv2.contourArea(contours[0])
        contourMaxAreaId = 0
        i = 0
        
        #para cada grupo de pixels branco
        for cnt in contours:
            #procura o grupo com a maior área
            if maxArea < cv2.contourArea(cnt):
                maxArea = cv2.contourArea(cnt)
                contourMaxAreaId = i
            i += 1

        approx = cv2.approxPolyDP(
          contours[contourMaxAreaId], 
          0.02 * cv2.arcLength(contours[contourMaxAreaId], True), 
          True
        )
            
        #achei o contorno com maior área em pixels
        cntMaxArea = contours[contourMaxAreaId]
    
    return frame, gray, maxArea, cntMaxArea

# Definir faixa de cores HSV para segmentação
# Triângulo: 101, 95, 179
# Círculo: 88, 77, 230
# Estrela: 40, 60, 157
# Hexagono: 0, 143, 208
# Quadrado: 98, 20, 23
# Casa: 0, 121, 153
# Cruz: 133, 43, 168
# Pentagono: 133, 105, 67

values = [
  {
    "hue_min": 82, "hue_max": 143, 
    "sat_min": 103, "sat_max": 255, 
    "val_min": 72, "val_max": 171
  },
  {
    "hue_min": 88, "hue_max": 255, 
    "sat_min": 77, "sat_max": 255, 
    "val_min": 230, "val_max": 255
  },
  {
    "hue_min": 40, "hue_max": 255, 
    "sat_min": 60, "sat_max": 255, 
    "val_min": 157, "val_max": 255
  },
  {
    "hue_min": 0, "hue_max": 255, 
    "sat_min": 143, "sat_max": 255, 
    "val_min": 208, "val_max": 255
  },
  {
    "hue_min": 98, "hue_max": 255, 
    "sat_min": 20, "sat_max": 255, 
    "val_min": 23, "val_max": 255
  },
  {
    "hue_min": 0, "hue_max": 255, 
    "sat_min": 121, "sat_max": 255, 
    "val_min": 153, "val_max": 255
  },
  {
    "hue_min": 133, "hue_max": 255, 
    "sat_min": 43, "sat_max": 255, 
    "val_min": 168, "val_max": 255
  },
  {
    "hue_min": 133, "hue_max": 255, 
    "sat_min": 105, "sat_max": 255, 
    "val_min": 67, "val_max": 255
  },
]

# Inicializa as trackbars com valores do SET_1
def setTrackbarValues(values):
    cv2.setTrackbarPos("Min Hue", trackbarWindow, values["hue_min"])
    cv2.setTrackbarPos("Max Hue", trackbarWindow, values["hue_max"])
    cv2.setTrackbarPos("Min Saturation", trackbarWindow, values["sat_min"])
    cv2.setTrackbarPos("Max Saturation", trackbarWindow, values["sat_max"])
    cv2.setTrackbarPos("Min Value", trackbarWindow, values["val_min"])
    cv2.setTrackbarPos("Max Value", trackbarWindow, values["val_max"])

trackbarWindow = "trackbar window"
cv2.namedWindow(trackbarWindow)

def onChange(val):
    return

cv2.createTrackbar("Min Hue", trackbarWindow, 0, 255, onChange)
cv2.createTrackbar("Max Hue", trackbarWindow, 255, 255, onChange)

cv2.createTrackbar("Min Saturation", trackbarWindow, 0, 255, onChange)
cv2.createTrackbar("Max Saturation", trackbarWindow, 255, 255, onChange)

cv2.createTrackbar("Min Value", trackbarWindow, 0, 255, onChange)
cv2.createTrackbar("Max Value", trackbarWindow, 255, 255, onChange)

min_hue = cv2.getTrackbarPos("Min Hue", trackbarWindow)
max_hue = cv2.getTrackbarPos("Max Hue", trackbarWindow)

min_sat = cv2.getTrackbarPos("Min Saturation", trackbarWindow)
max_sat = cv2.getTrackbarPos("Max Saturation", trackbarWindow)

min_val = cv2.getTrackbarPos("Min Value", trackbarWindow)
max_val = cv2.getTrackbarPos("Max Value", trackbarWindow)


debug = False

cap = cv2.VideoCapture(0)
# Cria uma única janela nomeada antes do loop
cv2.namedWindow('Segmentação', cv2.WINDOW_NORMAL)
cv2.namedWindow('Detecção de Formas', cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    allMaxArea = 0.0
    allFrame = None
    allGray = None
    allIndex = 0
    allCntMaxArea = None

    if debug:
      hue, sat, val = setLimitsOfTrackbar()
      frame, gray, maxArea = computeTracking(frame, hue, sat, val)

    else:
      for i in range(len(values[:2])):
        setTrackbarValues(values[i])
      
        hue, sat, val = setLimitsOfTrackbar()
        frame, gray, maxArea, cntMaxArea = computeTracking(frame, hue, sat, val)

        if allMaxArea <= maxArea:
          allMaxArea = maxArea

          allFrame = frame
          allGray = gray
          allIndex = i
          allCntMaxArea = cntMaxArea

      #retorna um retângulo que envolve o contorno em questão
      xRect, yRect, wRect, hRect = cv2.boundingRect(allCntMaxArea)

      #desenha caixa envolvente com espessura 3
      cv2.rectangle(
        allFrame, (xRect, yRect), 
        (xRect + wRect, yRect + hRect), 
        (0, 0, 255), 2
      )
      cv2.putText(
        allFrame, f"Index: {allIndex}", 
        (xRect, yRect - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 0, 255), 2
      )
    
    cv2.imshow("Segmentação", allGray)
    cv2.imshow("Detecção de Formas", allFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
