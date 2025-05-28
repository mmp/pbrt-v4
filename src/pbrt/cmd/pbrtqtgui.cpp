#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QTextEdit>
#include <QLabel>
#include <QFileDialog>
#include <QProcess> // Required for running pbrt
#include <QStringList> // For command line arguments

class PBRTMainWindow : public QMainWindow {
    Q_OBJECT

public:
    PBRTMainWindow(QWidget *parent = nullptr) : QMainWindow(parent), m_process(nullptr) {
        setWindowTitle("pbrt GUI Launcher");
        
        QWidget *centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);
        QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

        // --- Scene File Selection ---
        QHBoxLayout *sceneFileLayout = new QHBoxLayout();
        m_selectSceneButton = new QPushButton("Select Scene File");
        m_scenePathEdit = new QLineEdit();
        m_scenePathEdit->setReadOnly(true);
        m_scenePathEdit->setPlaceholderText("Path to scene file...");
        sceneFileLayout->addWidget(m_selectSceneButton);
        sceneFileLayout->addWidget(m_scenePathEdit);
        mainLayout->addLayout(sceneFileLayout);

        // --- Output File Path ---
        QHBoxLayout *outputFileLayout = new QHBoxLayout();
        m_outputFileLabel = new QLabel("Output File Path:");
        m_outputFilePathEdit = new QLineEdit("render.exr");
        outputFileLayout->addWidget(m_outputFileLabel);
        outputFileLayout->addWidget(m_outputFilePathEdit);
        mainLayout->addLayout(outputFileLayout);
        
        // --- Options ---
        QGridLayout *optionsLayout = new QGridLayout();
        m_interactiveCheckBox = new QCheckBox("Interactive Mode");
        m_useGPUCheckBox = new QCheckBox("Use GPU");
        m_threadsLabel = new QLabel("Threads:");
        m_threadsSpinBox = new QSpinBox();
        m_threadsSpinBox->setRange(1, 128); 
        m_threadsSpinBox->setValue(4);    

        optionsLayout->addWidget(m_interactiveCheckBox, 0, 0);
        optionsLayout->addWidget(m_useGPUCheckBox, 0, 1);
        optionsLayout->addWidget(m_threadsLabel, 1, 0);
        optionsLayout->addWidget(m_threadsSpinBox, 1, 1);
        mainLayout->addLayout(optionsLayout);

        // --- Render Button ---
        m_renderButton = new QPushButton("Start Render");
        mainLayout->addWidget(m_renderButton);

        // --- Console Output ---
        m_consoleLabel = new QLabel("Console Output:");
        m_consoleTextEdit = new QTextEdit();
        m_consoleTextEdit->setReadOnly(true);
        mainLayout->addWidget(m_consoleLabel);
        mainLayout->addWidget(m_consoleTextEdit);

        // --- Status Label ---
        m_statusLabel = new QLabel("Status: Idle");
        mainLayout->addWidget(m_statusLabel);
        
        resize(600, 500); // Increased height a bit for console

        // Connect signals and slots
        connect(m_selectSceneButton, &QPushButton::clicked, this, &PBRTMainWindow::onSelectSceneFile);
        connect(m_interactiveCheckBox, &QCheckBox::stateChanged, this, &PBRTMainWindow::onInteractiveChanged);
        connect(m_useGPUCheckBox, &QCheckBox::stateChanged, this, &PBRTMainWindow::onGPUChanged);
        connect(m_renderButton, &QPushButton::clicked, this, &PBRTMainWindow::onStartRender);

        // Set initial UI state based on checkboxes
        onInteractiveChanged(m_interactiveCheckBox->checkState());
        onGPUChanged(m_useGPUCheckBox->checkState());
    }

    ~PBRTMainWindow() {
        if (m_process) {
            if (m_process->state() != QProcess::NotRunning) {
                m_process->kill();
                m_process->waitForFinished(); // Wait for a bit, but don't block indefinitely
            }
            delete m_process;
            m_process = nullptr;
        }
    }

private slots:
    void onSelectSceneFile() {
        QString fileName = QFileDialog::getOpenFileName(this, 
                                                        "Select PBRT Scene File", 
                                                        "", 
                                                        "PBRT Scene Files (*.pbrt);;All Files (*)");
        if (!fileName.isEmpty()) {
            m_scenePathEdit->setText(fileName);
        }
    }

    void onInteractiveChanged(int state) {
        if (state == Qt::Checked) { // Interactive mode
            m_outputFilePathEdit->setEnabled(false);
            m_outputFilePathEdit->setText("N/A (Interactive)");
            m_threadsSpinBox->setEnabled(false); // Threads not typically set by user in interactive
        } else { // Batch mode
            m_outputFilePathEdit->setEnabled(true);
            m_outputFilePathEdit->setText("render.exr"); // Or restore a previous value if stored
            // GPU checkbox state now dictates threadsSpinBox state
            onGPUChanged(m_useGPUCheckBox->checkState()); 
        }
    }

    void onGPUChanged(int state) {
        if (!m_interactiveCheckBox->isChecked()) { // Only affect threads if not in interactive mode
            if (state == Qt::Checked) { // GPU is checked
                m_threadsSpinBox->setEnabled(false);
            } else { // GPU is unchecked
                m_threadsSpinBox->setEnabled(true);
            }
        }
    }

    void onStartRender() {
        if (m_scenePathEdit->text().isEmpty()) {
            m_statusLabel->setText("Status: Please select a scene file first.");
            return;
        }

        if (m_process && m_process->state() != QProcess::NotRunning) {
            m_statusLabel->setText("Status: Already rendering. Please wait or stop the current render.");
            return;
        }

        m_statusLabel->setText("Status: Rendering...");
        m_consoleTextEdit->clear();

        // Cleanup previous process if any (though usually handled by ~PBRTMainWindow or if it finished)
        delete m_process; 
        m_process = new QProcess(this);

        connect(m_process, &QProcess::readyReadStandardOutput, this, &PBRTMainWindow::onProcessOutput);
        connect(m_process, &QProcess::readyReadStandardError, this, &PBRTMainWindow::onProcessErrorOutput);
        connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &PBRTMainWindow::onProcessFinished);
        connect(m_process, &QProcess::errorOccurred, this, &PBRTMainWindow::onProcessLaunchError);

        QString pbrtExecutable = "pbrt"; // Assume pbrt is in PATH
        // On Windows, QProcess handles .exe suffix if it's in PATH.
        // If not in PATH, provide the full path to pbrt executable.

        QStringList arguments;
        arguments << m_scenePathEdit->text();

        if (m_interactiveCheckBox->isChecked()) {
            arguments << "--interactive";
            if (m_useGPUCheckBox->isChecked()) {
                arguments << "--gpu";
            } else {
                // As per instructions, default to wavefront if interactive and not GPU.
                // PBRT v4 might default to this, or might need explicit CPU backend.
                // For now, let's assume PBRT handles CPU by default in interactive if --gpu is not present.
                // Or use arguments << "--cpu"; or arguments << "--wavefront"; if needed
                arguments << "--wavefront"; 
            }
        } else { // Batch Mode
            arguments << "--outfile" << m_outputFilePathEdit->text();
            if (m_useGPUCheckBox->isChecked()) {
                arguments << "--gpu";
            } else {
                arguments << "--nthreads" << QString::number(m_threadsSpinBox->value());
            }
        }
        
        m_consoleTextEdit->append("Starting PBRT with command: " + pbrtExecutable + " " + arguments.join(" ") + "\n");
        m_process->start(pbrtExecutable, arguments);
    }

    void onProcessOutput() {
        m_consoleTextEdit->append(m_process->readAllStandardOutput());
    }

    void onProcessErrorOutput() {
        m_consoleTextEdit->append(m_process->readAllStandardError());
    }

    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
        if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            if (m_interactiveCheckBox->isChecked()) {
                m_statusLabel->setText("Status: Interactive session ended.");
            } else {
                m_statusLabel->setText("Status: Render Complete. Image saved to " + m_outputFilePathEdit->text());
            }
        } else {
            m_statusLabel->setText(QString("Status: Error - pbrt exited with code %1 (Status: %2)").arg(exitCode).arg(exitStatus));
            m_consoleTextEdit->append(QString("\nPBRT process exited with code %1.").arg(exitCode));
        }
    }

    void onProcessLaunchError(QProcess::ProcessError error) {
        QString errorMsg = m_process->errorString();
        m_statusLabel->setText(QString("Status: Error launching pbrt (%1)").arg(errorMsg));
        m_consoleTextEdit->append("Error launching pbrt: " + errorMsg);
        // Common errors: pbrt not found in PATH.
    }

private:
    // UI Elements
    QPushButton *m_selectSceneButton;
    QLineEdit   *m_scenePathEdit;
    QLabel      *m_outputFileLabel;
    QLineEdit   *m_outputFilePathEdit;
    QCheckBox   *m_interactiveCheckBox;
    QCheckBox   *m_useGPUCheckBox;
    QLabel      *m_threadsLabel;
    QSpinBox    *m_threadsSpinBox;
    QPushButton *m_renderButton;
    QLabel      *m_consoleLabel;
    QTextEdit   *m_consoleTextEdit;
    QLabel      *m_statusLabel;

    // PBRT Process
    QProcess *m_process;
};

#include "pbrtqtgui.moc" // Required for MOC compilation with Q_OBJECT

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    PBRTMainWindow mainWindow;
    mainWindow.show();
    return app.exec();
}
