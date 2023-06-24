from ultralytics import YOLO
import cv2
import math

def img_detection(path_x):

    model=YOLO("weights/best.pt")
    classNames = ['AppleFresh','AppleRotten','BananaRotten','BananaUnripe','DragonFruitFresh','DragonFruitRotten','GuavaFresh','GuavaRotten','MangoFresh','MangoRotten','MangoUnripe','OrangesFresh','OrangesRotten','OrangesUnripe','PapayaFresh','PapayaRotten','PapayaUnripe','PomegranateFresh','PomegranateRotten','StrawberryFresh','StrawberryUnripe','TomatoFresh','TomatoRotten','TomatoUnripe','BananaFresh','StrawberryRotten']

    success, img = cv2.imread(path_x)
    # success, img = cv.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=classNames[cls]
            label=f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

    return img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()