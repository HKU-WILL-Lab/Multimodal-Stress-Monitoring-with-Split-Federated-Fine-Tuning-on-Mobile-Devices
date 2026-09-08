plugins {
    id("com.android.application")
}

android {
    namespace = "org.mobihoc.wellbeing.wear"
    compileSdk = 37

    defaultConfig {
        applicationId = "org.mobihoc.wellbeing"
        minSdk = 33
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
    }

    flavorDimensions += "sensors"
    productFlavors {
        create("demo") {
            dimension = "sensors"
        }
        create("samsung") {
            dimension = "sensors"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":shared"))
    implementation("androidx.core:core-ktx:1.19.0")
    implementation("com.google.android.gms:play-services-wearable:20.0.1")
    add("samsungImplementation", files("libs/samsung-health-sensor-api.aar"))
}
