const phrase =
    "the quick brown fox jumps over the lazy dog 123";

let events = [];
let keyDownTime = {};

document.getElementById("typingBox").addEventListener("keydown", (e) => {
    keyDownTime[e.key] = performance.now();
});

document.getElementById("typingBox").addEventListener("keyup", (e) => {
    let up = performance.now();
    let down = keyDownTime[e.key] || up;

    events.push({
        key: e.key,
        hold: up - down,  // dwell time
        time: up
    });

    checkCompletion();
});

function checkCompletion() {
    let typed = document.getElementById("typingBox").value;

    if (typed === phrase) {
        document.getElementById("status").innerHTML = "Submitting...";

        fetch("/submit", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(events)
        })
        .then(res => res.text())
        .then(msg => {
            document.getElementById("status").innerHTML = msg;
        });

        events = [];
    }
}