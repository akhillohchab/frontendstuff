/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
var appf = {
    // Application Constructor
    initialize: function() {
        this.bindEvents();
    },
    // Bind Event Listeners
    //
    // Bind any events that are required on startup. Common events are:
    // 'load', 'deviceready', 'offline', and 'online'.
    bindEvents: function() {
        document.addEventListener('deviceready', this.onDeviceReady, false);
    },
    // deviceready Event Handler
    //
    // The scope of 'this' is the event. In order to call the 'receivedEvent'
    // function, we must explicitly call 'app.receivedEvent(...);'
    onDeviceReady: function() {
        //app.receivedEvent('deviceready');
    },
    // Update DOM on a Received Event
    receivedEvent: function(id) {
        var parentElement = document.getElementById(id);
        var listeningElement = parentElement.querySelector('.listening');
        var receivedElement = parentElement.querySelector('.received');

        listeningElement.setAttribute('style', 'display:none;');
        receivedElement.setAttribute('style', 'display:block;');

        console.log('Received Event: ' + id);
    },
	
	login: function() {
		//alert("Here");
		
		var fbDataSuccess = function(data) {
			alert(JSON.stringify(data));
			alert("name: " + data.name);
			name = data.name;
			localStorage.setItem("name", data.name)
			
			alert("name:" + localStorage.getItem("name"))
			alert("facebookID" + localStorage.getItem("facebookID"))
			alert("accessToken" + localStorage.getItem("accessToken"))
		}
		
		var fbPictureSuccess = function(data) {
			alert(JSON.stringify(data));
		}
	
		var fbLoginSuccess = function (userData) {
			//alert(JSON.stringify(userData.authResponse));
			//alert(userData.authResponse.accessToken)
			
			
			localStorage.setItem("facebookID", userData.authResponse.userID)
			localStorage.setItem("accessToken", userData.authResponse.accessToken)
			
			//var facebookID = userData.userID
			//var accessToken = userData.authResponse.accessToken
			
			
			facebookConnectPlugin.api('/me', ["public_profile"], fbDataSuccess, function(error) {alert(""+error)})

			//facebookConnectPlugin.api('/me/picture', ["public_profile"], fbPictureSuccess, function(error) {alert(""+error)})
			
			console.log(JSON.stringify(userData))
			
		}

		facebookConnectPlugin.login(["public_profile"],
			fbLoginSuccess,
			function (error) { alert("" + error) }
		);
	},
	
	logout: function() {
		var fbLogoutSuccess = function(data) {
			alert("Logged out")
		}
	
		facebookConnectPlugin.logout(fbLogoutSuccess,
			function (error) { console.log("" + error) }
		);
	
	}
	
	
};

appf.initialize();
