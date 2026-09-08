plugins {
    id("com.android.application")
}

val nativeSflEnabled = providers.gradleProperty("enableNativeSfl").orNull == "true"
val nativeSflLibDir = providers.gradleProperty("nativeSflLibDir")
    .orElse(file("../../sfl_runtime/build/android-toolchain/app-jni").absolutePath)
    .get()

android {
    namespace = "org.mobihoc.wellbeing.phone"
    compileSdk = 37

    defaultConfig {
        applicationId = "org.mobihoc.wellbeing"
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"

    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets.named("main") {
        if (nativeSflEnabled) {
            jniLibs.srcDir(nativeSflLibDir)
        }
    }
}

dependencies {
    implementation(project(":shared"))
    implementation("androidx.core:core-ktx:1.19.0")
    implementation("com.google.android.gms:play-services-wearable:20.0.1")
}
