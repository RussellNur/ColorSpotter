function setup() {

// Connect to the firebase
const firebaseConfig = {
  apiKey: "AIzaSyCnKqlroRmJgduXVfcgdfNDCNSmlp2HnUw",
  authDomain: "colorsclassifier.firebaseapp.com",
  databaseURL: "https://colorsclassifier.firebaseio.com",
  projectId: "colorsclassifier",
  storageBucket: "colorsclassifier.appspot.com",
  messagingSenderId: "183313961105",
  appId: "1:183313961105:web:f95fbadd22e4d1975eaf30"
};


createCanvas(400, 400);
  
// Declare global variables R, G, B
let r, g, b

// Set color function - which will be used when pic is updated
function setColor() {
  	r = floor(random(256));
  	g = floor(random(256));
  	b = floor(random(256));
}

setColor();
background(r, g, b);

// Create nine buttons, position them and assign the class for the CSS
let redButton = createButton('red');
redButton.position(5, 410)
redButton.class("RedButton")
let orangeButton = createButton('orange');
orangeButton.position(5, 440)
orangeButton.class("OrangeButton")
let yellowButton = createButton('yellow');
yellowButton.position(5, 470)
yellowButton.class("YellowButton")
let greenButton = createButton('green');
greenButton.position(5, 500)
greenButton.class("GreenButton")
let blueButton = createButton('blue');
blueButton.position(5, 530)
blueButton.class("BlueButton")
let purpleButton = createButton('purple');
purpleButton.position(5, 560)
purpleButton.class("PurpleButton")
let brownButton = createButton('brown');
brownButton.position(5, 590)
brownButton.class("BrownButton")
let grayButton = createButton('gray');
grayButton.position(5, 620)
grayButton.class("GrayButton")
let pinkButton = createButton('pink');
pinkButton.position(5, 650)
pinkButton.class("PinkButton")

// Send data to the firebase when the button is pressed
redButton.mousePressed(sendData);
orangeButton.mousePressed(sendData);
yellowButton.mousePressed(sendData);
greenButton.mousePressed(sendData);
blueButton.mousePressed(sendData);
purpleButton.mousePressed(sendData);
brownButton.mousePressed(sendData);
grayButton.mousePressed(sendData);
pinkButton.mousePressed(sendData);

// Initialize database
firebase.initializeApp(firebaseConfig);
// Create a database object
var database = firebase.database();

// A function to store the R, G, B values and the value itself in the firebase
function sendData() {
	// Create a reference to the colors section
	var colors = database.ref('colors');
	// Data object to be stored in the database
	var data = {
		red: r,
		green: g,
		blue: b,
		color: this.html()
	}
	
	// Push data to colors
	var color = colors.push(data);
	// Update the color of the page
	setColor();
	background(r, g, b);

	colors.once('value', function(snapshot) { console.log('Count: ' + snapshot.numChildren()); });
}


}
