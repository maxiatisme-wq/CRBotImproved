from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from train import *

# create the main window

mainwin = Tk()
mainwin.title("Clash Royale Bot")

# distinguish between whether the main program is running or not in the button to run the code and its associated text

executionButtonText = StringVar()
activeLabelText = StringVar()

executionButtonText.set("Run Clash Royale Bot")
activeLabelText.set("Clash Royale Bot is not running.")
crbot_active = False

def active():
    global executionButtonText, activeLabelText, crbot_active, programRunning
    if crbot_active == False:
        crbot_active = True
        executionButtonText.set("Stop Clash Royale Bot")
        activeLabelText.set("CRBot is running.")
        programRunning = True
        mainwin.destroy() # is there a way to implement this such that the UI keeps running while the program runs?
    else:
        executionButtonText.set("Run Clash Royale Bot")
        activeLabelText.set("CRBot is not running.")

# create the button to run the code and its associated text

executionButton = Button(mainwin, textvariable=executionButtonText, command=active)
executionButton.grid(row=0, column=0, columnspan=3, sticky=W)

activeLabel = Label(mainwin, textvariable=activeLabelText)
activeLabel.grid(row=1, column=0, columnspan=3, sticky=W)

# create buttons to select an output file, save to the output file selected by "Open", and save to a different file than that selected by "Open"

gameNumber = timePerformed = winLoss = "-"
openOutputFilePath = ""
def findOutputVariables(specifiedGameNumber=0):
    if openOutputFilePath != "":
        with open(openOutputFilePath, "r") as output:
            lineFound = False
            for line in output:
                global gameNumber, timePerformed, winLoss
                if line == f"Game number = {specifiedGameNumber}\n":
                    gameNumber = line[14:] # index 14 is the start of the value
                    gameNumber = gameNumber[:-1] # get rid of the newline
                    lineFound = True
                if "Time performed" in line and lineFound == True:
                    timePerformed = line[17:] # index 17 is the start of the value
                    timePerformed = timePerformed[:-1] # get rid of the newline
                if "Win/loss" in line and lineFound == True:
                    winLoss = line[11:] # index 11 is the start of the value
                    winLoss = winLoss[:-1] # get rid of the newline
                    lineFound = False
                if specifiedGameNumber == 0:
                    if "Game number" in line:
                        gameNumber = line[14:] # index 14 is the start of the value
                        gameNumber = gameNumber[:-1] # get rid of the newline
                    if "Time performed" in line:
                        timePerformed = line[17:] # index 17 is the start of the value
                        timePerformed = timePerformed[:-1] # get rid of the newline
                    if "Win/loss" in line:
                        winLoss = line[11:] # index 11 is the start of the value
                        winLoss = winLoss[:-1] # get rid of the newline
        global outputValues
        outputValues = {
            "Game number:" : gameNumber,
            "Time performed:" : timePerformed,
            "Win/loss:" : winLoss
        }
        getGameOutputValues().grid(row=8, column=0, columnspan=8, sticky=W)
def openOutputFile():
    global openOutputFilePath, openOutputFilePathLabelText
    testOpenOutputFilePath = filedialog.askopenfilename(initialdir="", title="Open File")
    # account for the user closing the dialog
    if testOpenOutputFilePath != "":
        openOutputFilePath = testOpenOutputFilePath
        openOutputFilePathLabelText.set(f"Open file: {openOutputFilePath}")
        findOutputVariables()
def saveAsOutputFile():
    global openOutputFilePath
    saveAsFilePath = filedialog.asksaveasfilename(initialdir="", title="Save File As...", filetypes=(("Text Files", "*.txt"), ("All Files", "*")))
    if saveAsFilePath != "" and openOutputFilePath != "":
        with open(saveAsFilePath, "w") as saveOutput:
            with open(openOutputFilePath, "r") as output:
                for line in output:
                    saveOutput.write(line)

openButton = Button(mainwin, text="Open Output File", command=openOutputFile)
openButton.grid(row=2, column=0, sticky=W)
saveAsButton = Button(mainwin, text="Save Open Output File As...", command=saveAsOutputFile)
saveAsButton.grid(row=3, column=0, sticky=W)

# display currently open file

openOutputFilePathLabelText = StringVar()
openOutputFilePathLabelText.set(f"Open file: (none)")
openOutputFilePathLabel = Label(mainwin, textvariable=openOutputFilePathLabelText)
openOutputFilePathLabel.grid(row=4, column=0, columnspan=8, sticky=W)

# list output values depending on the context

def getGameOutputValues():
    gameDataFrame = LabelFrame(mainwin, text="Game output values")
    counter = 0
    for key in outputValues:
        keyLabel = Label(gameDataFrame, text=key)
        keyLabel.grid(row=counter, column=1, sticky=W)
        itemLabel = Label(gameDataFrame, text=outputValues[key])
        itemLabel.grid(row=counter, column=2, sticky=W)
        counter += 1
    return gameDataFrame

# allow the user to retrieve the output of a previous game within the open file

openOtherGameOutputLabel = Label(mainwin, text="Enter previous game number within open file:")
openOtherGameOutputLabel.grid(row=5, column=0, columnspan=3, sticky=W)
openOtherGameOutputField = Entry(mainwin)
openOtherGameOutputField.grid(row=6, column=0, columnspan=1, sticky=W)
openOtherGameOutputButton = Button(mainwin, text="Go", command=lambda: findOutputVariables(openOtherGameOutputField.get()))
openOtherGameOutputButton.grid(row=7, column=0, columnspan=2, sticky=W)

mainwin.mainloop()

