from PySide6.QtWidgets import QApplication, QWidget
import sys

def main():
    app=QApplication(sys.argv)

    widget=QWidget()
    widget.resize(250,200)
    widget.move(1200,300)
    widget.setWindowTitle('Hello there')
    widget.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()