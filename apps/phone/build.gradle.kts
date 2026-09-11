plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
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
        compose = true
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
    implementation(platform("androidx.compose:compose-bom:2026.08.00"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.core:core-ktx:1.19.0")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("com.google.android.gms:play-services-wearable:20.0.1")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
