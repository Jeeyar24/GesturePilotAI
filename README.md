\# GesturePilot AI



\### Real-Time Hand Gesture Recognition and Contactless Computer Control



GesturePilot AI is an AI/ML project that recognizes hand gestures through a webcam and uses them to control PowerPoint presentations.



The project combines computer vision, hand landmark detection, machine learning, and a desktop dashboard.



\## Features



\- Real-time hand tracking using MediaPipe.

\- Custom hand gesture dataset.

\- Random Forest machine learning classifier.

\- Live gesture recognition and confidence display.

\- Gesture-based PowerPoint controls.

\- Interactive desktop dashboard.

\- Live camera preview.

\- Action history.

\- Start and Stop camera controls.

\- Gesture stability and command locking.



\## Supported Gestures



| Hand Gesture | Prediction | PowerPoint Action |

|---|---|---|

| Open palm | open\_palm | Next slide |

| Closed fist | fist | Previous slide |

| Peace sign | peace | Black screen |

| Thumbs up | thumbs\_up | White screen |



\## Technologies Used



\- Python

\- OpenCV

\- MediaPipe

\- Scikit-learn

\- Random Forest Classifier

\- Pandas

\- NumPy

\- Tkinter

\- PyAutoGUI

\- Pillow



\## Project Workflow



1\. Capture video using a webcam.

2\. Detect hand landmarks using MediaPipe.

3\. Extract numerical features from the landmarks.

4\. Predict gestures using a trained Random Forest classifier.

5\. Display the predicted gesture and confidence.

6\. Check gesture stability before performing an action.

7\. Send keyboard commands to PowerPoint when controls are enabled and the application is active.



\## Project Structure



\- `camera\_test.py` — Tests webcam access.

\- `hand\_test.py` — Detects hand landmarks.

\- `collect\_data.py` — Collects gesture training samples.

\- `gesture\_data.csv` — Stores the gesture dataset.

\- `train\_model.py` — Trains and evaluates the ML model.

\- `gesture\_model.joblib` — Stores the trained classifier.

\- `live\_test.py` — Recognizes gestures in real time.

\- `gesture\_control.py` — Controls PowerPoint using gestures.

\- `gesture\_dashboard.py` — Provides the desktop dashboard.

\- `hand\_landmarker.task` — MediaPipe hand landmark model.

\- `requirements.txt` — Lists Python dependencies.



\## Installation



\### Prerequisites



\- Windows computer

\- Python

\- Webcam

\- Microsoft PowerPoint for presentation controls



\### Setup



Open PowerShell in the project folder.



Create a virtual environment:



```powershell

py -m venv .venv

```



Install the required libraries:



```powershell

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```



Ensure the following files are present in the project folder:



\- `hand\_landmarker.task`

\- `gesture\_model.joblib`



\## Run the Application



Start the dashboard:



```powershell

.\\.venv\\Scripts\\python.exe gesture\_dashboard.py

```



Click \*\*Start Camera\*\* to begin hand tracking.



Enable PowerPoint commands only when you are ready to control a presentation.



\## Machine Learning Evaluation



The initial Random Forest model achieved 100% accuracy on a 30-sample held-out test split from the collected dataset.



This is an initial evaluation result, not a guarantee of real-world accuracy. Since the training and testing samples came from the same collection session, additional testing with new samples, different lighting conditions, and different hand positions is necessary.



\## Safety



\- Test the application with a practice presentation.

\- Enable PowerPoint controls only when needed.

\- Stop the camera when it is not in use.

\- Treat model confidence as an estimate, not a guarantee of correctness.



\## Future Improvements



\- Collect a larger and more diverse dataset.

\- Evaluate performance on new recording sessions.

\- Add customizable gestures and keyboard shortcuts.

\- Improve recognition under different lighting conditions.

\- Package the application for easier installation.



\## Conclusion



GesturePilot AI demonstrates how computer vision and supervised machine learning can enable contactless computer interaction.



The project was developed as a practical AI/ML learning project.

