# File upload

File must be a text file, comma deliniated, with headers energy and intensity.

```ts
const ping = await fetch("https://spectrum-classifier.fly.dev/api")
```

__API is online: ${ping.ok}__

Since the api is hosted on the free-tier, you might need to wait about 10 seconds for the 
first request to work as the server warms up, subsequenct requests should work fine.


```ts
const file = view(Inputs.file({ label: "Spectrum File", required: true }));
```

```ts

function extractLabel(filename: string): number | null {
    const match = filename.match(/_label_(\d+)_/);
    return match ? parseInt(match[1], 10) : null;
}

const expectedLabel = extractLabel(file.name)
const contents = await file.csv({ typed: true })

display("file read, hitting prediction api")
const response = await fetch("https://spectrum-classifier.fly.dev/api/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    energy: contents.map(c => c.energy),
    intensity: contents.map(c => c.intensity),
  })
});

if (response.ok) {
  const {
    prediction,
    main_peak_energy,
    peak_amplitudes,
    peak_centers
  } = await response.json()

  display(`Expected Label: ${expectedLabel}`)
  display(`Predicted Label: ${prediction}`)

  const peaks = peak_amplitudes.map((amplitude, index) => ({
      amplitude,
      center: peak_centers[index]
    }))

  display(
    Plot.plot({
      marks: [
        Plot.lineY(contents, {
          x: (d) => d.energy - main_peak_energy,
          y: 'intensity'
        }),
        Plot.dot(peaks, {
          x: 'center',
          y: 'amplitude'
        })
      ]
    })
  )
} else {
  display(await response.text())
}
```
