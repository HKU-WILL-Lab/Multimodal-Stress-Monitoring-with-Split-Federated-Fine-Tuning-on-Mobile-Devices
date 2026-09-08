package org.mobihoc.wellbeing.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AffectModelsTest {
    @Test
    fun mapsAllFiveUiStates() {
        assertEquals(AffectState.CALM, AffectMapper.map(5, 1))
        assertEquals(AffectState.ENERGIZED, AffectMapper.map(5, 5))
        assertEquals(AffectState.TENSE, AffectMapper.map(1, 5))
        assertEquals(AffectState.LOW_ENERGY, AffectMapper.map(1, 1))
        assertEquals(AffectState.NEUTRAL_MIXED, AffectMapper.map(3, 5))
    }

    @Test
    fun formatsNinetySecondTrend() {
        val outputs = listOf(
            ModelOutput(5, 1, "", 1),
            ModelOutput(3, 3, "", 2),
            ModelOutput(1, 5, "", 3),
        )
        assertTrue(TrendFormatter.format(outputs).contains("90 seconds"))
        assertTrue(TrendFormatter.format(outputs).contains("calm to tense"))
    }
}
