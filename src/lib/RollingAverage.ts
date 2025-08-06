export class RollingAverage {
    private buffer: number[]
    private size: number
    private head: number
    private tail: number
    private count: number
    private sum: number

    // Will initially fill with 0s
    // Until the buffer is full, the average will be wrong
    constructor(windowSize: number) {
        this.buffer = new Array(windowSize).fill(0)
        this.size = windowSize
        this.head = 0
        this.tail = 0
        this.count = 0
        this.sum = 0
    }

    push(value: number): void {
        if (this.count === this.size) {
            // Buffer is full, remove the oldest element
            this.sum -= this.buffer[this.head]
            this.head = (this.head + 1) % this.size
        } else {
            this.count++
        }

        // Add the new value
        this.buffer[this.tail] = value
        this.sum += value
        this.tail = (this.tail + 1) % this.size
    }

    getAverage(): number {
        if (this.count === 0) {
            return 0 // Avoid division by zero
        }
        return this.sum / this.count
    }

    getCount(): number {
        return this.count
    }
}
