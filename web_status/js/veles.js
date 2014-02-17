/*!
 * VELES web status interaction.
 * Copyright 2013 Samsung Electronics
 * Licensed under Samsung Proprietary License.
 */

function updateUI() {
	console.log("timer");
}

$(window).load(function() {
	setInterval(updateUI, 1000)
});