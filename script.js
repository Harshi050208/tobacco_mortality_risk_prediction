function predict() {
    const data = {
        age: document.getElementById("age").value,
        cigarettes: document.getElementById("cigarettes").value,
        years: document.getElementById("years").value,
        income: document.getElementById("income").value,
        disease: document.getElementById("disease").value
    };

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        document.getElementById("result").innerText = result.prediction;
    });
}
