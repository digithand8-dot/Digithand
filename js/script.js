function loginUser(event){

event.preventDefault();

let username = document.getElementById("username").value;
let password = document.getElementById("password").value;

if(username === "admin" && password === "1234"){
    
    window.location.href = "dashboard.html";

}else{

    alert("Invalid Username or Password");

}

}


function uploadImage(){

let file = document.getElementById("imageUpload").files[0];

if(!file){

document.getElementById("message").innerHTML = "Please select an image";
return;

}

document.getElementById("message").innerHTML = "Image uploaded successfully (Processing will be added later)";

}

function previewImage(){

let file = document.getElementById("imageUpload").files[0];

if(file){

let reader = new FileReader();

reader.onload = function(e){

document.getElementById("preview").src = e.target.result;

}

reader.readAsDataURL(file);

}

}