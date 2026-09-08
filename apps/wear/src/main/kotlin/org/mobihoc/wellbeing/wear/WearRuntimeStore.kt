package org.mobihoc.wellbeing.wear

data class WearRuntimeState(
    val collecting: Boolean = false,
    val status: String = "Ready",
    val currentState: String = "No estimate yet",
    val suggestion: String = "",
)

object WearRuntimeStore {
    @Volatile
    var state = WearRuntimeState()
        private set

    @Synchronized
    fun update(transform: (WearRuntimeState) -> WearRuntimeState) {
        state = transform(state)
    }
}
