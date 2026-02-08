import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (32, 32))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {"input": img})
    pred = np.argmax(outputs[0])

    cv2.putText(
        frame,
        f"Pred: {pred}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Edge AI Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
