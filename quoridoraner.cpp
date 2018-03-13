// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include <math.h>

// for all others, include the necessary headers (this file is usually all you
// need because it includes almost all "standard" wxWidgets headers)
#ifndef WX_PRECOMP
    #include "wx/wx.h"
#endif

#include "Game.h"


class MyApp : public wxApp
{
public:
    virtual bool OnInit();
};

// Define a new frame type: this is going to be our main frame
class MyFrame : public wxFrame
{
public:
    // ctor(s)
    MyFrame(const wxChar *title, int xpos, int ypos, int width, int height);

    // event handlers (these functions should _not_ be virtual)
    void OnQuit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
	void OnPaint(wxPaintEvent & event);
	void OnTopButton(wxCommandEvent& WXUNUSED(event));
	void OnRightButton(wxCommandEvent& WXUNUSED(event));
	void OnBotButton(wxCommandEvent& WXUNUSED(event));
	void OnLeftButton(wxCommandEvent& WXUNUSED(event));
	void OnMouseClick(wxMouseEvent & event);
	void OnHorizontallyButton(wxCommandEvent& WXUNUSED(event));
	void OnVerticalButton(wxCommandEvent& WXUNUSED(event));
	void OnComputerMove(wxCommandEvent& WXUNUSED(event));
    void OnRedo(wxCommandEvent & event);
    void OnUndo(wxCommandEvent & event);

private:
    // any class wishing to process wxWidgets events must use this macro
    wxDECLARE_EVENT_TABLE();

	Game game;
	wxButton *topButton, *rightButton, *botButton, *leftButton;
	wxButton *horizontButton, *vertButton;
	wxButton *computerMoveButton;
	wxTextCtrl *textBox;

	bool vertButtonClicked = false, horizontButtonClicked = false;
};

// IDs for the controls and the menu commands
enum
{
	// menu items
	Minimal_Quit = wxID_EXIT,
	Minimal_About = wxID_ABOUT,
    ID_UNDO,
    ID_REDO,
	TextBox
};

//Button IDs:
enum {
	IDTop = wxID_HIGHEST+1,
IDRight,
IDLeft,
IDBot,
IDVertical,
IDHorizontally,
IDComputerMove
};

// ----------------------------------------------------------------------------
// event tables and other macros for wxWidgets
// ----------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
	EVT_MENU(Minimal_Quit, MyFrame::OnQuit)
	EVT_MENU(Minimal_About, MyFrame::OnAbout)
    EVT_MENU(ID_REDO, MyFrame::OnRedo)
    EVT_MENU(ID_UNDO, MyFrame::OnUndo)
	EVT_PAINT(MyFrame::OnPaint)
	EVT_BUTTON(IDTop,MyFrame::OnTopButton)
	EVT_BUTTON(IDRight,MyFrame::OnRightButton)
	EVT_BUTTON(IDBot, MyFrame::OnBotButton)
	EVT_BUTTON(IDLeft, MyFrame::OnLeftButton)
	EVT_BUTTON(IDHorizontally, MyFrame::OnHorizontallyButton)
	EVT_BUTTON(IDVertical,MyFrame::OnVerticalButton)
	EVT_BUTTON(IDComputerMove,MyFrame::OnComputerMove)
	EVT_LEFT_DOWN(MyFrame::OnMouseClick)
wxEND_EVENT_TABLE()


IMPLEMENT_APP(MyApp)

// 'Main program' equivalent: the program execution "starts" here
bool MyApp::OnInit()
{
    // call the base class initialization method, currently it only parses a
    // few common command-line options but it could be do more in the future
    if ( !wxApp::OnInit() )
        return false;

    // create the main application window
    MyFrame *frame = new MyFrame(wxT("Quoridoraner"),100,100,1400,800);

    // and show it (the frames, unlike simple controls, are not shown when
    // created initially)
    frame->Show(true);

    return true;
}



// frame constructor
MyFrame::MyFrame(const wxChar *title, int xpos, int ypos, int width, int height)
	: wxFrame((wxFrame *)NULL, -1, title, wxPoint(xpos, ypos), wxSize(width, height))
{
    

    // create a menu bar
    wxMenu *fileMenu = new wxMenu;

    // the "About" item should be in the help menu
    wxMenu *helpMenu = new wxMenu;
    helpMenu->Append(Minimal_About, "&About\tF1", "Show about dialog");


    fileMenu->Append(Minimal_Quit, "E&xit\tAlt-X", "Quit this program");
    fileMenu->Append(ID_UNDO, "Undo", "Undo the last move");
    fileMenu->Append(ID_REDO, "Redo", "Redo the last move");
    // now append the freshly created menu to the menu bar...
    wxMenuBar *menuBar = new wxMenuBar();
    menuBar->Append(fileMenu, "&File");
    menuBar->Append(helpMenu, "&Help");

    // ... and attach this menu bar to the frame
    SetMenuBar(menuBar);

	// Place Buttons:
    topButton = new wxButton(this, IDTop, wxT("Top"), wxPoint(800, 40),wxSize(40,100));
	botButton = new wxButton(this, IDBot, wxT("Bot"), wxPoint(800, 180), wxSize(40, 100));

	leftButton = new wxButton(this, IDLeft, wxT("Left"), wxPoint(800-100, 140),wxSize(100,40));
	rightButton = new wxButton(this, IDRight, wxT("Right"), wxPoint(800+40, 140), wxSize(100, 40));

	horizontButton = new wxButton(this, IDHorizontally, wxT(""), wxPoint(1000, 160), wxSize(120, 30));
	vertButton = new wxButton(this, IDVertical, wxT(""), wxPoint(1200, 80), wxSize(30, 120));

	computerMoveButton = new wxButton(this, IDComputerMove, wxT("Computer Move"), wxPoint(800, 300), wxSize(100, 40));

	textBox = new wxTextCtrl(this, TextBox, wxT("Ready for plays"), wxPoint(700, 400), wxSize(500, 200),wxTE_READONLY| wxTE_MULTILINE | wxTE_RICH);

}


// event handlers

void MyFrame::OnQuit(wxCommandEvent& WXUNUSED(event))
{
    // true is to force the frame to close
    Close(true);
}

void MyFrame::OnAbout(wxCommandEvent& WXUNUSED(event))
{
    wxMessageBox(wxString::Format
                 (
                    "Welcome to %s!\n"
                    "\n"
                    "This is the Quoridoraner\n"
                    "running under %s.",
                    wxVERSION_STRING,
                    wxGetOsDescription()
                 ),
                 "About Quoridoraner",
                 wxOK | wxICON_INFORMATION,
                 this);
}

void MyFrame::OnPaint(wxPaintEvent& event) {
	wxPaintDC dc(this);
	game.drawGame(&dc);

	textBox->Clear();
	*textBox << game.msg;

	
}


void MyFrame::OnTopButton(wxCommandEvent& WXUNUSED(event)) {
    game.tryMoving(0);
	Refresh();
}

void MyFrame::OnRightButton(wxCommandEvent& WXUNUSED(event))
{
    game.tryMoving(1);
	Refresh();
}

void MyFrame::OnBotButton(wxCommandEvent& WXUNUSED(event)) {
    game.tryMoving(2);
	Refresh();
}
void MyFrame::OnLeftButton(wxCommandEvent& WXUNUSED(event)) {
    game.tryMoving(3);
	Refresh();
}

void MyFrame::OnMouseClick(wxMouseEvent& event) {

	
	wxClientDC dc(this);
	wxPoint p(event.GetLogicalPosition(dc));
    int x = round((double)(p.x - 45) / 70); // Calculate clicking position
	int y = round((double)(p.y + 25) / 70) - 2;
	if (vertButtonClicked) {
		game.tryBarrier(Barrier(x,y, true));
		vertButtonClicked = false;
	}
	else if (horizontButtonClicked) {
		game.tryBarrier(Barrier(x,y, false));
		horizontButtonClicked = false;
	}
    *textBox << "Set Barrier to: " << x << "|" << y <<  "\n" << p.x << "|" << p.y << "\n";

	Refresh();
}

void MyFrame::OnHorizontallyButton(wxCommandEvent & event)
{
	horizontButtonClicked = !horizontButtonClicked;
	textBox->Clear();
	*textBox << "Horizontal Barrier Placement\n";
	
}

void MyFrame::OnVerticalButton(wxCommandEvent & event)
{
	vertButtonClicked = !vertButtonClicked;
	textBox->Clear();
	*textBox << "Vertical Barrier Placement\n";
	
}


void MyFrame::OnComputerMove(wxCommandEvent & event) {
	game.computerMove();
	Refresh();
}

void MyFrame::OnRedo(wxCommandEvent & event){// TODO: Fill with functionality

}
void MyFrame::OnUndo(wxCommandEvent & event){// TODO: Fill with functionality

}
