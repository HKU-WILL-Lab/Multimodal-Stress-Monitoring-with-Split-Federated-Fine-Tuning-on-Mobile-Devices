package org.mobihoc.wellbeing.phone

import android.app.Application

/** Owns the headless phone runtime independently of any visual interface. */
class MobiWellbeingApplication : Application() {
    lateinit var wellbeingController: WellbeingController
        private set

    override fun onCreate() {
        super.onCreate()
        wellbeingController = WellbeingController(applicationContext)
    }

    override fun onTerminate() {
        wellbeingController.close()
        super.onTerminate()
    }
}
