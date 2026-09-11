package org.mobihoc.wellbeing.phone

import android.app.Application

/** Owns one shared runtime across the phone UI and incoming watch messages. */
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
